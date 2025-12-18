import { Server } from './server'
import { Server as SocketIoServer } from 'socket.io'

export class Messenger {
  server: Server
  io: SocketIoServer

  constructor (server: Server) {
    console.log('messenger')
    this.io = new SocketIoServer(server.httpServer)
    this.server = server
    this.setupIo()
  }

  setupIo (): void {
    this.io.on('connection', socket => {
      console.log(socket.id, 'connected')
      socket.on('disconnect', () => {
        console.log(socket.id, 'disconnected')
      })
    })
  }
}
