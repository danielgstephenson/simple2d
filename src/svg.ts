import { readFileSync } from 'fs-extra'
import { INode, parseSync } from 'svgson'
import path from 'path'
import { pointsOnPath } from 'points-on-path'
import { getDistance } from './math'

export function parseSvg(fileName: string): INode {
  const svgPath = path.join(__dirname, 'resources', fileName)
  const svgString = readFileSync(svgPath, 'utf8')
  return parseSync(svgString)
}

export function getChildById(svgNode: INode, id: string): INode {
  for (const child of svgNode.children) {
    if (child.attributes.id !== id) continue
    return child
  }
  throw new Error(`Child ${id} not found`)
}

export function getChildrenByRole(svgNode: INode, role: string): INode[] {
  const children = svgNode.children.filter(child => child.attributes.role === role)
  return children
}

export function flip(point: number[]): number[] {
  return [point[0], -point[1]]
}

export function getPathPoints(pathNode: INode): number[][] {
  const points = pointsOnPath(pathNode.attributes.d).flat()
  const distance = getDistance(points[0], points[points.length - 1])
  if (distance === 0) points.pop()
  const flipped = points.map(flip)
  return flipped
}

export function getCircleCenter(circleNode: INode): number[] {
  const x = Number(circleNode.attributes.cx)
  const y = Number(circleNode.attributes.cy)
  return flip([x, y])
}

export function getCircleRadius(circleNode: INode): number {
  const radius = Number(circleNode.attributes.r)
  return radius
}