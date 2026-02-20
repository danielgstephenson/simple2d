import { cp, mkdir } from "fs/promises"
import { dirname } from "path"

async function main() {
  const from = "src/resources"
  const to = "dist/resources"

  // Ensure parent directory exists
  await mkdir(dirname(to), { recursive: true })

  // Copy directory recursively, overwrite existing files
  await cp(from, to, { recursive: true, force: true })

  console.log(`Copied ${from} -> ${to}`)
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
});