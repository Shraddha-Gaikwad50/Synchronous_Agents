import { existsSync } from "node:fs";
import { NextRequest, NextResponse } from "next/server";
import { v2 } from "@google-cloud/speech";

export const runtime = "nodejs";

/** Max time for the serverless function (e.g. Vercel); adjust if needed. */
export const maxDuration = 120;

/**
 * GET: safe config snapshot for debugging voice (no secrets).
 * Open http://127.0.0.1:3000/api/transcribe in the browser while dev is running.
 */
export async function GET() {
  const path = process.env.GOOGLE_APPLICATION_CREDENTIALS?.trim() ?? "";
  return NextResponse.json({
    projectId: process.env.GOOGLE_CLOUD_PROJECT?.trim() ?? null,
    hasInlineServiceAccountJson: Boolean(
      process.env.GCP_SERVICE_ACCOUNT_JSON?.trim()
    ),
    googleApplicationCredentialsPath: path || null,
    credentialsFileExists: path ? existsSync(path) : false,
    speechLocation: process.env.GCP_SPEECH_LOCATION?.trim() || "us",
    speechModel: process.env.GCP_SPEECH_MODEL?.trim() || "chirp_3",
    languageCode: process.env.GCP_SPEECH_LANGUAGE_CODE?.trim() || "en-US",
    speechApiEndpoint:
      speechApiEndpointForLocation(
        process.env.GCP_SPEECH_LOCATION?.trim() || "us"
      ) ?? "speech.googleapis.com (global default)",
  });
}

type RecognizeResponse = {
  results?: {
    alternatives?: { transcript?: string | null }[] | null;
  }[] | null;
};

function extractTranscript(response: RecognizeResponse): string {
  const parts: string[] = [];
  for (const r of response.results ?? []) {
    for (const alt of r.alternatives ?? []) {
      const t = alt.transcript;
      if (t) parts.push(t);
    }
  }
  return parts.join(" ").trim();
}

function gcpErrorMessage(err: unknown): string {
  if (err && typeof err === "object" && "message" in err) {
    return String((err as { message: unknown }).message);
  }
  return String(err);
}

/** Regional recognizers (e.g. `us`, `eu`, `us-central1`) must use `{location}-speech.googleapis.com`. */
function speechApiEndpointForLocation(location: string): string | undefined {
  const loc = location.trim().toLowerCase();
  if (!loc || loc === "global") return undefined;
  return `${loc}-speech.googleapis.com`;
}

export async function POST(request: NextRequest) {
  let formData: FormData;
  try {
    formData = await request.formData();
  } catch {
    return NextResponse.json({ error: "Invalid multipart body." }, { status: 400 });
  }

  const file = formData.get("file");
  if (!file || typeof file === "string") {
    return NextResponse.json(
      { error: 'Missing audio file field "file".' },
      { status: 400 }
    );
  }

  const rawSa = process.env.GCP_SERVICE_ACCOUNT_JSON?.trim();
  let sa: Record<string, unknown> | undefined;
  if (rawSa) {
    try {
      sa = JSON.parse(rawSa) as Record<string, unknown>;
    } catch {
      return NextResponse.json(
        { error: "GCP_SERVICE_ACCOUNT_JSON is set but is not valid JSON." },
        { status: 503 }
      );
    }
  }

  const location = process.env.GCP_SPEECH_LOCATION?.trim() || "us";
  const model = process.env.GCP_SPEECH_MODEL?.trim() || "chirp_3";
  const languageCode = process.env.GCP_SPEECH_LANGUAGE_CODE?.trim() || "en-US";
  const apiEndpoint = speechApiEndpointForLocation(location);

  const clientOpts: ConstructorParameters<typeof v2.SpeechClient>[0] = {};
  if (sa) {
    clientOpts.credentials = sa as NonNullable<
      ConstructorParameters<typeof v2.SpeechClient>[0]
    >["credentials"];
  }
  if (apiEndpoint) {
    clientOpts.apiEndpoint = apiEndpoint;
  }

  let client: v2.SpeechClient;
  try {
    client = new v2.SpeechClient(clientOpts);
  } catch (e) {
    return NextResponse.json(
      { error: `Invalid GCP credentials: ${gcpErrorMessage(e)}` },
      { status: 503 }
    );
  }

  if (sa && !sa.project_id && !process.env.GOOGLE_CLOUD_PROJECT?.trim()) {
    return NextResponse.json(
      {
        error:
          "Service account JSON is missing project_id. Set GOOGLE_CLOUD_PROJECT to your GCP project ID.",
      },
      { status: 503 }
    );
  }

  const projectId =
    process.env.GOOGLE_CLOUD_PROJECT?.trim() ||
    process.env.GCP_PROJECT_ID?.trim() ||
    (typeof sa?.project_id === "string" ? sa.project_id : "") ||
    (await client.getProjectId());

  if (!projectId) {
    return NextResponse.json(
      {
        error:
          "Could not determine GCP project ID. Set GOOGLE_CLOUD_PROJECT or use a service account JSON that includes project_id.",
      },
      { status: 503 }
    );
  }

  const buffer = Buffer.from(await file.arrayBuffer());
  if (buffer.length < 256) {
    return NextResponse.json({ error: "Audio file too small." }, { status: 400 });
  }

  const recognizer = client.recognizerPath(projectId, location, "_");

  try {
    const [response] = await client.recognize({
      recognizer,
      config: {
        autoDecodingConfig: {},
        model,
        languageCodes: [languageCode],
        features: {
          enableAutomaticPunctuation: true,
        },
      },
      content: buffer,
    });

    const text = extractTranscript(response);
    return NextResponse.json({ text });
  } catch (e) {
    const msg = gcpErrorMessage(e);
    const isClient =
      /invalid argument|audio|too long|duration|exceeds|empty/i.test(msg);
    return NextResponse.json(
      { error: msg || "Speech-to-Text request failed." },
      { status: isClient ? 400 : 502 }
    );
  }
}
