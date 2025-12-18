export const pi = Math.PI

export const twoPi = 2 * pi

export const X = 0
export const Y = 1

export function range (a: number, b?: number): number[] {
  if (b == null) return range(0, a - 1)
  return [...Array(b - a + 1).keys()].map(i => a + i)
}

export function sample<T> (options: T[]): T {
  return options[Math.floor(Math.random() * options.length)]
}

export function sum (array: number[]): number {
  let total = 0
  array.forEach(x => { total = total + x })
  return total
}

export function average (array: number[]): number {
  if (array.length === 0) return 0
  return sum(array) / array.length
}

export function dot (x: number[], y: number[]): number {
  if (x.length !== y.length) {
    throw new Error('x and y must have the same length')
  }
  const products = x.map((_, i) => x[i] * y[i])
  return sum(products)
}

export function clamp (a: number, b: number, x: number): number {
  return Math.max(a, Math.min(x, b))
}

export function vecToAngle (vector: number[]): number {
  const x = vector[0]
  const y = vector[1]
  return Math.atan2(y, x)
}

export function angleToDir (angle: number): number[] {
  return [Math.cos(angle), Math.sin(angle)]
}

export function sub (a: number[], b: number[]): number[] {
  return range(a.length).map(i => a[i] - b[i])
}

export function add (a: number[], b: number[]): number[] {
  return range(a.length).map(i => a[i] + b[i])
}

export function combine (a: number, v: number[], b: number, w: number[]): number[] {
  return add(mul(a, v), mul(b, w))
}

export function mul (a: number, v: number[]): number[] {
  return v.map(x => a * x)
}

export function getLength (v: number[]): number {
  const squares = v.map(x => x ** 2)
  return Math.sqrt(sum(squares))
}

export function normalize (v: number[]): number[] {
  const length = getLength(v)
  if (length === 0) return [0, 0]
  return mul(1 / length, v)
}

export function dirToFrom (to: number[], from: number[]): number[] {
  return normalize(sub(from, to))
}

export function dirFromTo (from: number[], to: number[]): number[] {
  return normalize(sub(to, from))
}

export function whichMax (array: number[]): number {
  let indexMax = 0
  let valueMax = array[0]
  array.forEach((value, index) => {
    if (value > valueMax) {
      indexMax = index
      valueMax = value
    }
  })
  return indexMax
}

export function whichMin (array: number[]): number {
  const negArray = array.map(x => -x)
  return whichMax(negArray)
}

// export function getAngleDiff (toAngle: number, fromAngle: number): number {
//   const v = { x: Math.cos(fromAngle), y: Math.sin(fromAngle) }
//   const w = { x: Math.cos(toAngle), y: Math.sin(toAngle) }
//   return Math.atan2(w.y * v.x - w.x * v.y, w.x * v.x + w.y * v.y)
// }

// export function rotate (vector: Vec2, angle: number): Vec2 {
//   const x = vector.x * Math.cos(angle) - vector.y * Math.sin(angle)
//   const y = vector.x * Math.sin(angle) + vector.y * Math.cos(angle)
//   return new Vec2(x, y)
// }

// export function randomDir (): Vec2 {
//   return rotate(new Vec2(1, 0), 2 * Math.PI * Math.random())
// }

// export function round (x: number, digits: number): number {
//   return parseFloat(x.toFixed(digits))
// }

// export function roundDir (dir: Vec2): Vec2 {
//   const compassDirs = range(0, 7).map(i => rotate(new Vec2(0, 1), 2 * Math.PI * i / 8))
//   if (dir.length() === 0) return dir
//   const dots = compassDirs.map(compassDir => Vec2.dot(compassDir, dir))
//   return compassDirs[whichMax(dots)]
// }

// export function clampVec (vector: Vec2, maxLength: number): Vec2 {
//   const length = vector.length()
//   if (length < maxLength) return vector
//   const direction = normalize(vector)
//   return Vec2.mul(direction, maxLength)
// }

// export function project (a: Vec2, b: Vec2): Vec2 {
//   const ab = Vec2.dot(a, b)
//   const bb = Vec2.dot(b, b)
//   if (ab === 0 || bb === 0) return Vec2(0, 0)
//   return Vec2.mul(ab / bb, b)
// }

// export function reject (a: Vec2, b: Vec2): Vec2 {
//   return Vec2.sub(a, project(a, b))
// }
