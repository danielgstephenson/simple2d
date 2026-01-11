import { combine, cross, range, sub } from './math'

export function raycastSegment (rayStart: number[], rayVector: number[], segmentStart: number[], segmentEnd: number[]): number[][] {
  const intersections: number[][] = []
  const segmentVector = sub(segmentEnd, segmentStart)
  const startDifference = sub(segmentStart, rayStart)
  const denominator = cross(rayVector, segmentVector)
  if (denominator === 0) return intersections
  const rayFactor = cross(startDifference, segmentVector) / denominator
  if (rayFactor < 0) return intersections
  const segmentFactor = cross(startDifference, rayVector) / denominator
  if (segmentFactor < 0) return intersections
  if (segmentFactor > 1) return intersections
  const intersection = combine(1, rayStart, rayFactor, rayVector)
  intersections.push(intersection)
  return intersections
}

export function raycastWall (rayStart: number[], rayVector: number[], wall: number[][]): number[][] {
  const intersections: number[][] = []
  range(wall.length).forEach(i => {
    const j = i > 0 ? i - 1 : wall.length - 1
    const segmentIntersections = raycastSegment(rayStart, rayVector, wall[i], wall[j])
    intersections.push(...segmentIntersections)
  })
  return intersections
}
