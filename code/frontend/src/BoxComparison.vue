<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'

const props = defineProps({ models: { type: Array, required: true } })
const result = ref(null)
const index = ref(null)
const loading = ref(false)
const error = ref('')
let request

const runIds = computed(() => props.models.map(model => model.id).join(','))
const examples = computed(() => result.value ? [
  { id: 'expected', name: '人工标注', boxes: result.value.expected.map(xyxy => ({ xyxy })) },
  ...props.models.map(model => ({ ...model, boxes: result.value.predictions[model.id] ?? [] })),
] : [])

async function showImage(nextIndex) {
  request?.abort()
  const controller = new AbortController()
  request = controller
  loading.value = true
  error.value = ''
  try {
    const params = new URLSearchParams(props.models.map(model => ['run', model.id]))
    if (nextIndex) params.set('index', nextIndex)
    const response = await fetch(`/api/box-comparison?${params}`, { signal: controller.signal })
    const answer = await response.json()
    if (!response.ok) throw new Error(answer.detail ?? '无法读取测试图')
    if (request === controller) {
      result.value = answer
      index.value = answer.index
    }
  } catch (failure) {
    if (failure.name !== 'AbortError') error.value = failure.message
  } finally {
    if (request === controller) loading.value = false
  }
}

function position([left, top, right, bottom]) {
  const { width, height } = result.value
  return {
    left: `${left * 100 / width}%`,
    top: `${top * 100 / height}%`,
    width: `${(right - left) * 100 / width}%`,
    height: `${(bottom - top) * 100 / height}%`,
  }
}

watch(runIds, () => {
  result.value = null
  showImage()
}, { immediate: true })

onBeforeUnmount(() => request?.abort())
</script>

<template>
  <section class="panel">
    <header class="panel-head"><h2>同一张测试图的画框对照</h2></header>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p>
    <p v-if="loading && !result" class="caption" role="status">正在读取测试图…</p>
    <template v-if="result">
      <form class="tools example-navigation" @submit.prevent="showImage(index)">
        <button type="button" class="button small" :disabled="loading || result.index === 1" @click="showImage(result.index - 1)">上一张</button>
        <label>测试图 <input v-model.number="index" type="number" min="1" :max="result.total" required :disabled="loading"> / {{ result.total }}</label>
        <button class="button small" :disabled="loading">查看</button>
        <button type="button" class="button small" :disabled="loading || result.index === result.total" @click="showImage(result.index + 1)">下一张</button>
      </form>
      <div class="example-grid">
        <figure v-for="example in examples" :key="example.id">
          <figcaption>{{ example.name }} <small>{{ example.boxes.length }} 个框</small></figcaption>
          <div class="location-stage example-stage" :style="{ aspectRatio: `${result.width} / ${result.height}` }">
            <img :src="result.url" :alt="`${example.name} · 同一张蜜蜂测试图`">
            <span v-for="(box, order) in example.boxes" :key="order" class="example-box"
              :class="{ 'example-expected': example.id === 'expected' }" :style="position(box.xyxy)"
              :aria-label="box.confidence == null ? '人工标注框' : `预测框，置信度 ${(box.confidence * 100).toFixed(1)}%`" />
          </div>
        </figure>
      </div>
    </template>
  </section>
</template>

<style scoped>
.example-navigation { flex-wrap: wrap; margin-bottom: 14px; }
.example-navigation input { width: 76px; padding: 5px; }
.example-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(0, 1fr)); gap: 10px 14px; }
.example-grid figure { display: grid; grid-template-rows: subgrid; grid-row: span 2; margin: 0; }
.example-grid figcaption { font-weight: 650; }
.example-grid small { margin-left: 5px; font-weight: 400; color: var(--muted); }
.example-stage { width: 100%; margin: 0; }
.example-box { position: absolute; border: 2px solid #facc15; box-shadow: 0 0 0 1px #291e09; }
.example-expected { border-color: #28e58a; }
@media (max-width: 1200px) { .example-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 650px) { .example-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
</style>
