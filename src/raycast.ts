import { combine, cross, getDistance, range, sub } from './math'

export function rayCastSegment(rayStart: number[], rayVector: number[], segment: number[][]): number[][] {
  const segmentStart = segment[0]
  const segmentEnd = segment[1]
  const segmentVector = sub(segmentEnd, segmentStart)
  const startDifference = sub(segmentStart, rayStart)
  const denominator = cross(rayVector, segmentVector)
  if (denominator === 0) return []
  const rayFactor = cross(startDifference, segmentVector) / denominator
  if (rayFactor < 0) return []
  const segmentFactor = cross(startDifference, rayVector) / denominator
  if (segmentFactor < 0) return []
  if (segmentFactor > 1) return []
  const intersection = combine(1, rayStart, rayFactor, rayVector)
  return [intersection]
}

export function rayCastWall(rayStart: number[], rayVector: number[], wall: number[][]): number[][] {
  const intersections: number[][] = []
  range(wall.length).forEach(i => {
    const j = i > 0 ? i - 1 : wall.length - 1
    const segment = [wall[i], wall[j]]
    const segmentIntersections = rayCastSegment(rayStart, rayVector, segment)
    intersections.push(...segmentIntersections)
  })
  return intersections
}

export function insideWall(point: number[], wall: number[][]): boolean {
  const intersections = rayCastWall(point, [1, 0], wall)
  return (intersections.length % 2) === 1
}

export function segmentCastSegment(a: number[][], b: number[][]): number[][] {
  const aStart = a[0]
  const aEnd = a[1]
  const bStart = b[0]
  const bEnd = b[1]
  const aVector = sub(aEnd, aStart)
  const bVector = sub(bEnd, bStart)
  const startDifference = sub(aStart, bStart)
  const denominator = cross(bVector, aVector)
  if (denominator === 0) return []
  const aFactor = cross(startDifference, bVector) / denominator
  if (aFactor < 0) return []
  if (aFactor > 1) return []
  const bFactor = cross(startDifference, aVector) / denominator
  if (bFactor < 0) return []
  if (bFactor > 1) return []
  const intersection = combine(1, aStart, aFactor, aVector)
  return [intersection]
}

export function pointDistSegment(point: number[], segment: number[][]): number {
  const segmentVector = sub(segment[1], segment[0])
  const perp0 = [+segmentVector[1], -segmentVector[0]]
  const perp1 = [-segmentVector[1], +segmentVector[0]]
  const intersections0 = rayCastSegment(point, perp0, segment)
  const intersections1 = rayCastSegment(point, perp1, segment)
  const intersections = [...intersections0, ...intersections1]
  const distances = intersections.map(intersection => getDistance(point, intersection))
  distances.push(getDistance(point, segment[0]))
  distances.push(getDistance(point, segment[1]))
  return Math.min(...distances)
}

export function pointDistWall(point: number[], wall: number[][]): number {
  let minDistance = Infinity
  range(wall.length).forEach(i => {
    const j = i > 0 ? i - 1 : wall.length - 1
    const segment = [wall[i], wall[j]]
    const distance = pointDistSegment(point, segment)
    minDistance = Math.min(minDistance, distance)
  })
  return minDistance
}
