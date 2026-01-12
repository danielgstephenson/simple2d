import { combine, cross, range, sub } from './math'

export function rayCastSegment (rayStart: number[], rayVector: number[], segmentStart: number[], segmentEnd: number[]): number[][] {
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

export function rayCastWall (rayStart: number[], rayVector: number[], wall: number[][]): number[][] {
  const intersections: number[][] = []
  range(wall.length).forEach(i => {
    const j = i > 0 ? i - 1 : wall.length - 1
    const segmentIntersections = rayCastSegment(rayStart, rayVector, wall[i], wall[j])
    intersections.push(...segmentIntersections)
  })
  return intersections
}

export function insideWall (point: number[], wall: number[][]): boolean {
  const intersections = rayCastWall(point, [1, 0], wall)
  return (intersections.length % 2) === 1
}

export function segmentCastSegment (aStart: number[], aEnd: number[], bStart: number[], bEnd: number[]): number[][] {
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
