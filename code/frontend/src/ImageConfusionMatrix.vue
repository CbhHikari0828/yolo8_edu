<script setup>
import { computed } from 'vue'
import ChartPanel from './ChartPanel.vue'

const props = defineProps({
  title: { type: String, required: true },
  filename: { type: String, required: true },
  matrix: { type: Array, required: true },
  max: { type: Number, default: 0 },
})

const classes = ['有螨', '无螨']
const options = computed(() => ({
  animation: false,
  grid: { left: 80, right: 74, top: 34, bottom: 78, containLabel: true },
  xAxis: { type: 'category', data: classes, name: '预测标签', nameLocation: 'middle', nameGap: 35 },
  yAxis: { type: 'category', data: classes, name: '真实标签', nameLocation: 'middle', nameGap: 45, inverse: true },
  tooltip: { formatter: ({ value: [predicted, actual, count] }) => `真实${classes[actual]}，预测${classes[predicted]}，${count} 张` },
  visualMap: {
    min: 0, max: props.max || Math.max(1, ...props.matrix.flat()), orient: 'vertical', right: 10, top: 'middle',
    inRange: { color: ['#e8f4f1', '#14786b'] },
  },
  series: [{
    type: 'heatmap', data: props.matrix.flatMap((row, actual) => row.map((count, predicted) => [predicted, actual, count])),
    label: { show: true, color: '#111d1a', textBorderColor: '#f2faf8', textBorderWidth: 2, fontSize: 17 },
  }],
}))
</script>

<template>
  <ChartPanel :title="title" :filename="filename" :options="options" />
</template>
