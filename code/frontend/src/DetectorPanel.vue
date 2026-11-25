<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'

const emit = defineEmits(['open-run'])
const model = ref(null)
const preview = ref('')
const filename = ref('')
const result = ref(null)
const error = ref('')
const detecting = ref(false)
let activeRequest

function percentage(value) {
  return `${(value * 100).toFixed(1)}%`
}

async function loadStatus() {
  try {
    const response = await fetch('/api/detection')
    if (!response.ok) throw new Error('无法读取检测模型状态。')
    model.value = await response.json()
  } catch (failure) {
    error.value = failure.message
  }
}

async function detect(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file) return
  error.value = ''
  if (file.size > 10 * 1024 * 1024) {
    error.value = '请上传不超过 10 MB 的照片。'
    return
  }
  if (preview.value) URL.revokeObjectURL(preview.value)
  preview.value = URL.createObjectURL(file)
  filename.value = file.name
  result.value = null
  detecting.value = true
  activeRequest = new AbortController()
  try {
    const response = await fetch('/api/detection', {
      method: 'POST',
      body: file,
      headers: { 'Content-Type': file.type || 'application/octet-stream' },
      signal: activeRequest.signal,
    })
    const answer = await response.json().catch(() => null)
    if (!response.ok || !answer) throw new Error(answer?.detail ?? '检测失败，请换一张照片再试。')
    result.value = answer
    if (answer.source_run !== model.value?.source_run) await loadStatus()
  } catch (failure) {
    if (failure.name !== 'AbortError') error.value = failure.message
  } finally {
    detecting.value = false
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

onMounted(loadStatus)
onBeforeUnmount(() => {
  activeRequest?.abort()
  if (preview.value) URL.revokeObjectURL(preview.value)
})
</script>

<template>
  <div class="grid two locations-grid">
    <section class="panel">
      <header class="panel-head"><h2>上传蜜蜂照片</h2><span v-if="model?.ready" class="status completed">检测模型就绪</span></header>
      <label class="location-input">选择一张照片开始检测
        <input type="file" aria-label="上传照片检测" accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"
          :disabled="detecting || !model?.ready" @change="detect">
      </label>
      <p class="caption">PNG、JPG 或 WebP，最大 10 MB。请使用单只蜜蜂清晰的近照。</p>
      <p v-if="model && !model.ready" class="notice">检测模型准备中。<button class="button small" @click="loadStatus">刷新状态</button></p>
      <p v-if="detecting" class="notice" role="status">正在检测照片…</p>
      <p v-if="error" class="notice error" role="alert">{{ error }}</p>
      <template v-if="model?.test">
        <header class="panel-head detector-model"><h2>检测模型测试结果</h2></header>
        <p class="caption">当前模型 {{ model.model }} · {{ model.source_run }}</p>
        <button class="button small" @click="emit('open-run', model.source_run)">查看此模型的训练记录</button>
        <p class="caption">VarroaDataset 官方测试集 · {{ model.test.image.samples.toLocaleString('zh-CN') }} 张图片</p>
        <table><tbody>
          <tr><th>单图有无判断准确率</th><td>{{ percentage(model.test.image.accuracy) }}</td></tr>
          <tr><th>有螨图片召回率</th><td>{{ percentage(model.test.image.mite_recall) }}</td></tr>
          <tr><th>位置检测 mAP@0.50</th><td>{{ percentage(model.test.map50) }}</td></tr>
        </tbody></table>
      </template>
    </section>

    <section class="panel">
      <header class="panel-head"><h2>瓦螨检测结果</h2></header>
      <template v-if="result">
        <p :class="['notice', { 'detected-mite': result.boxes.length }]" role="status">
          <strong>{{ result.boxes.length ? `检测到 ${result.boxes.length} 处疑似瓦螨` : '未检测到可见瓦螨' }}</strong>
        </p>
        <p v-if="!result.boxes.length" class="caption">可能受拍摄角度或光线影响，可以换一张蜂体更清晰的近照再试。</p>
        <div class="location-stage detector-stage" :style="{ aspectRatio: `${result.width} / ${result.height}`, width: `min(100%, 560px, ${500 * result.width / result.height}px)` }">
          <img :src="preview" alt="用户上传的蜜蜂照片">
          <span v-for="(box, index) in result.boxes" :key="index" class="mite-marker detector-marker"
            :style="position(box.xyxy)" :aria-label="`疑似瓦螨 ${index + 1}，置信度 ${percentage(box.confidence)}`">
            <span>{{ index + 1 }}</span>
          </span>
        </div>
        <div v-if="result.boxes.length" class="tools detector-scores">
          <span v-for="(box, index) in result.boxes" :key="index" class="status">{{ index + 1 }} · {{ percentage(box.confidence) }}</span>
        </div>
        <p class="sample-path">{{ filename }}</p>
      </template>
      <img v-else-if="preview" class="detector-preview" :src="preview" alt="用户上传的蜜蜂照片">
      <p v-else class="caption">选择照片后，检测到的位置会用黄色圈标出。</p>
    </section>
  </div>
</template>
