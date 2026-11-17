export async function api(path, options = {}) {
  const response = await fetch("/api" + path, options);
  if (!response.ok) {
    let body;
    try {
      body = await response.json();
    } catch {}
    const error = new Error(
      typeof body?.detail === "string"
        ? body.detail
        : `Yêu cầu thất bại (${response.status})`,
    );
    error.status = response.status;
    throw error;
  }
  return response.json();
}
export const json = (method, body) => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
export function socket(path) {
  return new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`,
  );
}
