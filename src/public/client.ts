import { io } from 'socket.io-client'
import { Renderer } from '../renderer'
import { Input } from '../input'
import { dot, getLength, whichMax } from '../math'
import { actionVectors } from '../actionVectors'
import { Layout, LevelSummary } from '../world/level'

export class Client {
  renderer = new Renderer()
  input = new Input()
  socket = io()
  token = 0

  constructor() {
    window.addEventListener('keydown', (event: KeyboardEvent) => {
      if (!event.repeat) this.socket.emit('unpause')
    })
    this.setupIo()
    setInterval(() => this.sendAction(), 20)
  }

  setupIo() {
    this.socket.on('connect', () => {
      console.log('connect')
    })
    this.socket.on('token', (token: number) => {
      if (this.token !== 0 && this.token !== token) {
        console.log('reload', this.token, token)
        location.reload()
        return
      }
      this.token = token
    })
    this.socket.on('renderScale', (renderScale: number) => {
      this.renderer.renderScale = renderScale
    })
    this.socket.on('summary', (summary: LevelSummary) => {
      this.renderer.summary = summary
    })
    this.socket.on('layout', (layout: Layout) => {
      this.renderer.layout = layout
    })
  }

  sendAction(): void {
    this.renderer.camera.updateScale(this.input.zoom)
    let x = 0
    let y = 0
    if (this.input.isKeyDown('KeyW') || this.input.isKeyDown('ArrowUp')) y += 1
    if (this.input.isKeyDown('KeyS') || this.input.isKeyDown('ArrowDown')) y -= 1
    if (this.input.isKeyDown('KeyA') || this.input.isKeyDown('ArrowLeft')) x -= 1
    if (this.input.isKeyDown('KeyD') || this.input.isKeyDown('ArrowRight')) x += 1
    const vector = [x, y]
    if (getLength(vector) === 0) {
      this.socket.emit('action', 0)
      return
    }
    const dots = actionVectors.map(dir => dot(dir, vector))
    const action = whichMax(dots)
    this.socket.emit('action', action)
  }
}

