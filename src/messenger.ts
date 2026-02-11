import { Server } from './server'
import { Server as SocketIoServer } from 'socket.io'
import { Level } from './world/level'

export class Messenger {
  server: Server
  io: SocketIoServer
  level: Level
  token = Math.random()

  constructor(server: Server) {
    this.io = new SocketIoServer(server.httpServer)
    this.server = server
    this.level = new Level()
    this.setupIo()
    setInterval(() => this.update(), 20)
    // console.log('svg json:')
    // console.log(JSON.stringify(svgObject, null, 2))
  }

  setupIo(): void {
    this.io.on('connection', socket => {
      console.log(socket.id, 'connected')
      socket.emit('token', this.token)
      socket.emit('renderScale', this.server.config.renderScale)
      socket.emit('layout', this.level.layout())
      socket.on('action', (action: number) => {
        if (this.level.agents.length > 0) {
          this.level.agents[0].action = action
        }
      })
      socket.on('unpause', () => {
        this.level.paused = false
      })
      socket.on('disconnect', () => {
        console.log(socket.id, 'disconnected')
      })
    })
  }

  update(): void {
    this.io.emit('summary', this.level.summary)
  }
}
