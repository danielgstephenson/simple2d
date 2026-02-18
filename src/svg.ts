import { readFileSync } from 'fs-extra'
import { createSVGWindow } from 'svgdom'
import { Element, SVG, registerWindow } from '@svgdotjs/svg.js'
import path from 'path'
import { pointsOnPath } from 'points-on-path'
import { getDistance } from './math'

const window = createSVGWindow()
const document = window.document
registerWindow(window, document)

export function getSvg(fileName: string): Element {
  const filePath = path.join(__dirname, 'resources', fileName)
  const svgString = readFileSync(filePath, 'utf8')
  const element = SVG(svgString)
  return element
}

export function flipPoint(point: number[]): number[] {
  return [point[0], -point[1]]
}

export function getPathPoints(element: Element): number[][] {
  const pathString = element.attr('d')
  if (typeof pathString !== 'string') {
    throw new Error('typeof pathString !== "string"')
  }
  const points = pointsOnPath(pathString).flat().map(p => flipPoint(p))
  const distance = getDistance(points[0], points[points.length - 1])
  if (distance === 0) points.pop()
  return points
}

export function getCenter(element: Element): number[] {
  return flipPoint([element.cx(), element.cy()])
}

export function getRadius(element: Element): number {
  const radius = element.attr('r')
  if (typeof radius !== 'number') {
    throw new Error("typeof radius !== 'number'")
  }
  return radius
}

export function findElement(element: Element, selector: string): Element {
  const child = element.findOne(selector)
  if (!(child instanceof Element)) {
    throw new Error('element not found')
  }
  return child
}

export function findElements(element: Element, selector: string): Element[] {
  return [...element.find(selector)]
}