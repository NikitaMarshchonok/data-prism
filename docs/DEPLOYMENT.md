# Deployment and operations

This guide describes the repository's current single-instance deployment shape. It does not claim multi-region or high-availability operation.

## Render Blueprint

The root `render.yaml` defines a Docker web service with:

- automatic deployment only after linked CI checks pass;
- `/readyz` as the deployment health check;
- generated session and monitoring API secrets;
- one Gunicorn worker with four threads for the 512 MB free plan;
- a 25 MB public upload limit and bounded VibeDash previews;
- structured JSON application logs.

To deploy:

1. Sign in to Render and select **New → Blueprint**.
2. Connect `NikitaMarshchonok/data-prism`.
3. Review the proposed free web service and apply the Blueprint.
4. Wait for the Docker build, CI-gated deploy, and readiness check.
5. Open the assigned `onrender.com` URL and run the built-in VibeDash demo.

Render documents the current Blueprint workflow and schema at:

- <https://render.com/docs/infrastructure-as-code>
- <https://render.com/docs/blueprint-spec>

No external AI key is required for the deterministic demo. `OPENAI_API_KEY` can be added later as a secret if optional narrative summaries are needed.

## Persistence boundary

The free Render service uses an ephemeral filesystem. Uploaded datasets, reports, baselines, and drift history are therefore lost when the instance is restarted or redeployed. This is acceptable for a public portfolio demo, but not for persistent monitoring.

For single-instance persistent monitoring, upgrade to a paid service and attach a disk at:

```text
/var/lib/data-prism
```

Keep `DATA_PRISM_STATE_DIR=/var/lib/data-prism`. Render's disk documentation explains the cost and operational constraints: <https://render.com/docs/disks>.

A persistent disk restricts the service to one instance and prevents zero-downtime deploys. A future multi-instance architecture should instead move uploads and reports to object storage and drift history to PostgreSQL.

## Runtime signals

Each response includes an `X-Request-ID`. A valid inbound request ID is preserved; otherwise the application creates a new one. Application request logs contain:

- UTC timestamp and log level;
- service version or deployment commit;
- request ID;
- HTTP method and normalized Flask route;
- response status and duration in milliseconds.

Raw request bodies, uploaded filenames, query strings, IP addresses, and session identifiers are not added to access logs.

Render supplies `RENDER_GIT_COMMIT` automatically. Data Prism exposes its shortened value in `/healthz` and `/readyz` and includes it in structured request logs.

## Operational verification

```bash
curl -i https://YOUR-SERVICE.onrender.com/healthz
curl -i https://YOUR-SERVICE.onrender.com/readyz
curl -i -H "X-Request-ID: portfolio-check-001" \
  https://YOUR-SERVICE.onrender.com/vibedash/
```

Verify that:

- both health endpoints return HTTP 200;
- the response includes the same safe request ID;
- the JSON log event contains the normalized route rather than a raw URL;
- the reported deployment version matches the active Render commit.

The pull-request workflow also builds the production Docker image, starts it with generated test credentials, waits for `/readyz`, verifies request-ID propagation, and confirms that a structured request event reached container logs. This provides the container-level verification even when a contributor does not run Docker locally.

## Rollback

If a deployment fails its readiness check, Render keeps the previous successful version active. For an application-level regression after deployment, select the last known-good deploy in the Render dashboard and redeploy it, then confirm `/readyz` before resuming traffic.
