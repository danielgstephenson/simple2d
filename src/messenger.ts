import { Server } from './server'
import { Server as SocketIoServer } from 'socket.io'
import { World } from './world/world'
import { Testbed } from './world/testbed'
import { svgObject } from './entities/level'

export class Messenger {
  server: Server
  io: SocketIoServer
  world: World

  constructor (server: Server) {
    console.log('messenger')
    this.io = new SocketIoServer(server.httpServer)
    this.server = server
    this.world = new Testbed()
    this.setupIo()
    setInterval(() => this.update(), 20)
    console.log('svg json:')
    console.log(JSON.stringify(svgObject, null, 2))
  }

  setupIo (): void {
    this.io.on('connection', socket => {
      console.log(socket.id, 'connected')
      socket.emit('renderScale', this.server.config.renderScale)
      socket.on('action', (action: number) => {
        if (this.world.agents.length > 0) {
          this.world.agents[0].action = action
        }
      })
      socket.on('unpause', () => {
        this.world.paused = false
      })
      socket.on('disconnect', () => {
        console.log(socket.id, 'disconnected')
      })
    })
  }

  update (): void {
    this.io.emit('summary', this.world.summary)
  }
}
