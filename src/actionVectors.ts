import { angleToDir, range, twoPi } from './math'

export const actionVectors: number[][] = []

actionVectors.push([0, 0])

range(8).forEach(i => {
  const angle = twoPi * i / 8
  const dir = angleToDir(angle)
  actionVectors.push(dir)
})
