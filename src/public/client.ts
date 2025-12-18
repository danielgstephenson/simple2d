import { io } from 'socket.io-client'
import { Renderer } from '../renderer'
import { WorldSummary } from '../world/world'
import { Input } from '../input'
import { dot, getLength, whichMax } from '../math'
import { actionVectors } from '../actionVectors'

const renderer = new Renderer()
const input = new Input()

function sendAction (): void {
  renderer.camera.updateScale(input.zoom)
  let x = 0
  let y = 0
  if (input.isKeyDown('KeyW') || input.isKeyDown('ArrowUp')) y += 1
  if (input.isKeyDown('KeyS') || input.isKeyDown('ArrowDown')) y -= 1
  if (input.isKeyDown('KeyA') || input.isKeyDown('ArrowLeft')) x -= 1
  if (input.isKeyDown('KeyD') || input.isKeyDown('ArrowRight')) x += 1
  const vector = [x, y]
  if (getLength(vector) === 0) {
    socket.emit('action', 0)
    return
  }
  const dots = actionVectors.map(dir => dot(dir, vector))
  const action = whichMax(dots)
  socket.emit('action', action)
}

setInterval(sendAction, 20)

const socket = io()
socket.on('connect', () => {
  console.log('connect')
})
socket.on('renderScale', (renderScale: number) => {
  renderer.renderScale = renderScale
})
socket.on('summary', (summary: WorldSummary) => {
  renderer.summary = summary
})
