export async function requestImage(body, signal) {
  const response = await fetch("/api/images",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body),signal});
  const result = await response.json().catch(()=>({}));
  if (!response.ok) throw new Error(result.error || "The image service is unavailable. Your original design is kept.");
  if (!/^data:image\/(?:png|webp|jpeg);base64,[a-z0-9+/=]+$/i.test(result.image || "")) throw new Error("The image service did not return a usable image.");
  return result.image;
}
