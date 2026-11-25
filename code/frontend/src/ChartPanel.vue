<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts/core'
import { BarChart, HeatmapChart, LineChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent, VisualMapComponent } from 'echarts/components'
import { CanvasRenderer, SVGRenderer } from 'echarts/renderers'

echarts.use([
  BarChart, HeatmapChart, LineChart, GridComponent, LegendComponent,
  TooltipComponent, VisualMapComponent, CanvasRenderer, SVGRenderer,
])

const props = defineProps({
  title: { type: String, required: true },
  options: { type: Object, required: true },
  filename: { type: String, required: true },
})
const element = ref(null)
let chart
let observer

function draw(instance) {
  instance.setOption({
    color: ['#0e766e', '#bf7623', '#325ba1', '#894880'],
    textStyle: { fontFamily: 'system-ui, Microsoft YaHei, sans-serif', color: '#536461' },
    backgroundColor: '#ffffff',
    ...props.options,
  }, { notMerge: true })
}

onMounted(() => {
  chart = echarts.init(element.value, null, { renderer: 'svg' })
  draw(chart)
  observer = new ResizeObserver(() => chart?.resize())
  observer.observe(element.value)
})
watch(() => props.options, () => { if (chart) draw(chart) })
onBeforeUnmount(() => {
  observer?.disconnect()
  chart?.dispose()
})

function exportChart(type) {
  const host = document.createElement('div')
  host.style.cssText = 'position:fixed;left:-10000px;top:0;width:960px;height:540px'
  document.body.append(host)
  const exportInstance = echarts.init(host, null, { renderer: type === 'svg' ? 'svg' : 'canvas' })
  try {
    draw(exportInstance)
    exportInstance.getZr().flush()
    const link = document.createElement('a')
    link.download = `${props.filename}.${type}`
    link.href = exportInstance.getDataURL({ type, pixelRatio: 2, backgroundColor: '#ffffff' })
    link.click()
  } finally {
    exportInstance.dispose()
    host.remove()
  }
}
</script>

<template>
  <section class="panel chart-panel">
    <header class="panel-head">
      <h2>{{ title }}</h2>
      <div class="tools" aria-label="导出图表">
        <button class="button small" @click="exportChart('png')">PNG</button>
        <button class="button small" @click="exportChart('svg')">SVG</button>
      </div>
    </header>
    <div ref="element" class="chart" role="img" :aria-label="title"></div>
  </section>
</template>
