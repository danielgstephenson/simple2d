import fs from 'fs-extra'
import path from 'path'
import * as svgson from 'svgson'

const svgPath = path.resolve(__dirname, '..', 'resources', 'test.svg')
export const svgString = fs.readFileSync(svgPath, 'utf8')

export const svgObject = svgson.parseSync(svgString)
