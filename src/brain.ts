
import { range, whichMax } from './math'
import * as ort from 'onnxruntime-node'
import { Imagination } from './world/imagination'

export class Brain {
  imagination = new Imagination()
  session?: ort.InferenceSession
  busy: boolean = false
  action: number = 0

  constructor () {
    void this.startSession()
  }

  async startSession (): Promise<void> {
    this.session = await ort.InferenceSession.create('./model/onnx/value.onnx')
    const state0 = [-3, 0, 0, 0, -3, 0, 0, 0]
    const state1 = [+3, 0, 0, 0, +3, 0, 0, 0]
    const state = [...state0, ...state1]
    for (const i of range(4)) {
      console.time(`inference ${i}`)
      const outcomes = this.imagination.getOutcomes(state)
      const data = Float32Array.from(outcomes.flat())
      const tensor = new ort.Tensor('float32', data, [81, 16])
      const feeds = { state: tensor }
      const result = await this.session.run(feeds)
      console.timeEnd(`inference ${i}`)
      if (!(result.output.data instanceof Float32Array)) return
      const values = Array.from(result.output.data)
      // values.forEach(value => {
      //   console.log(value.toFixed(4))
      // })
      const valueMatrix: number[][] = []
      range(9).forEach(r => {
        valueMatrix[r] = range(9 * r, 9 * r + 8).map(i => values[i])
      })
      // valueMatrix[0].forEach(value => {
      //   console.log(value.toFixed(4))
      // })
      const actionValues = valueMatrix.map(row => Math.min(...row))
      void actionValues
      // actionValues.forEach(actionValue => {
      //   console.log(actionValue.toFixed(4))
      // })
    }
  }

  async update (state: number[]): Promise<void> {
    if (this.busy) return
    if (this.session == null) return
    this.busy = true
    // const outcomes = this.imagination.getOutcomes(state)
    const outcomes = [[
      -1.5383503606070938, 1.6556488490003125,
      -1.2060559371513624, 0.5890940403668103,
      -1.4174889692338488, 2.4769011919216943,
      -3.1537424456779806, 0.9168236340224539,
      -2.690579221153769, 0,
      1.349066778281722, 0,
      -2.9942584965125607, 0,
      0.04695504633279549, 0
    ]]
    const data = Float32Array.from(outcomes.flat())
    const tensor = new ort.Tensor('float32', data, [outcomes.length, 16])
    const feeds = { state: tensor }
    const result = await this.session.run(feeds)
    if (!(result.output.data instanceof Float32Array)) return
    const modelValues = Array.from(result.output.data)
    const values = outcomes.map(outcome => {
      const position = [outcome[0], outcome[1]]
      const distance = Math.sqrt(position[0] ** 2 + position[1] ** 2)
      return -distance
    })
    const n = values.length
    const sqErrors = range(n).map(i => (values[i] - modelValues[i]) ** 2)
    console.log('sqErrors', sqErrors)
    console.log(values, modelValues)
    const valueMatrix: number[][] = []
    range(9).forEach(r => {
      valueMatrix[r] = range(9 * r, 9 * r + 8).map(i => modelValues[i])
    })
    const actionValues = valueMatrix.map(row => Math.min(...row))
    this.action = whichMax(actionValues)
    this.busy = false
  }
}
