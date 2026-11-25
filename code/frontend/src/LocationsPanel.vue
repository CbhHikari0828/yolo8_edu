<script setup>
import { onMounted, ref } from 'vue'

const category = ref('mite')
const query = ref('')
const samples = ref([])
const selected = ref(null)
const listError = ref('')
const lookupError = ref('')
const loading = ref(false)
const matching = ref(false)
const showMarks = ref(true)
let selection = 0

async function fetchLocation(path) {
  const response = await fetch(`/api/locations/${path}`)
  const result = await response.json()
  if (!response.ok) {
    throw new Error(typeof result.detail === 'string' ? result.detail : '暂时无法读取训练图片。')
  }
  return result
}

async function findSamples() {
  loading.value = true
  listError.value = ''
  try {
    const params = new URLSearchParams({ category: category.value, q: query.value })
    samples.value = await fetchLocation(`samples?${params}`)
    if (!selected.value && !matching.value && samples.value.length) {
      selectSample(samples.value[0])
    }
  } catch (error) {
    listError.value = error.message
    samples.value = []
  } finally {
    loading.value = false
  }
}

function selectSample(sample) {
  selection++
  selected.value = sample
  lookupError.value = ''
  matching.value = false
  showMarks.value = true
}

async function matchImage(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file) return
  const request = ++selection
  matching.value = true
  lookupError.value = ''
  try {
    if (!globalThis.crypto?.subtle) throw new Error('请从本机的 http://127.0.0.1:8088 打开页面再选择图片。')
    const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer())
    const sha256 = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')
    const result = await fetchLocation(`lookup?${new URLSearchParams({ sha256, name: file.name })}`)
    if (selection === request) selectSample(result)
  } catch (error) {
    if (selection === request) lookupError.value = error.message
  } finally {
    if (selection === request) matching.value = false
  }
}

function position([left, top, right, bottom], sample) {
  return {
    left: `${left * 100 / sample.width}%`,
    top: `${top * 100 / sample.height}%`,
    width: `${(right - left) * 100 / sample.width}%`,
    height: `${(bottom - top) * 100 / sample.height}%`,
  }
}

onMounted(findSamples)
</script>

<template>
  <div class="grid two locations-grid">
    <section class="panel">
      <header class="panel-head"><h2>选择训练照片</h2></header>
      <label class="location-input">从电脑选择训练集原图
        <input type="file" accept=".png,image/png" @change="matchImage">
      </label>
      <p v-if="matching" class="caption" role="status">正在匹配训练集原图…</p>
      <p v-if="lookupError" class="notice error" role="alert">{{ lookupError }}</p>
      <div class="location-samples">
        <h2>训练集样本</h2>
        <form class="location-search" @submit.prevent="findSamples">
          <select v-model="category" aria-label="样本类别" @change="findSamples">
            <option value="mite">有可见瓦螨</option><option value="no_mite">无可见瓦螨</option>
          </select>
          <input v-model.trim="query" type="search" maxlength="160" aria-label="筛选样本" placeholder="输入文件名或 bee_id">
          <button class="button small" :disabled="loading">查找</button>
        </form>
        <p v-if="listError" class="notice error" role="alert">{{ listError }}</p>
        <p v-else-if="!loading && !samples.length" class="caption">没有找到符合条件的训练图。</p>
        <div class="location-list" v-else>
          <button v-for="sample in samples" :key="sample.id" class="location-item"
            :class="{ active: selected?.id === sample.id }" :aria-pressed="selected?.id === sample.id"
            @click="selectSample(sample)">
            <img :src="`/api/locations/images/${sample.id}`" width="36" height="53" loading="lazy" alt="">
            <span>{{ sample.name }}</span>
          </button>
        </div>
      </div>
    </section>

    <section class="panel">
      <header class="panel-head">
        <h2>原图与位置标注</h2>
        <label v-if="selected?.boxes.length" class="location-toggle"><input v-model="showMarks" type="checkbox">显示标注圈</label>
      </header>
      <template v-if="selected">
        <div class="location-stage" :style="{ aspectRatio: `${selected.width} / ${selected.height}` }">
          <img :src="`/api/locations/images/${selected.id}`" :alt="`训练集原图 ${selected.name}`">
          <template v-if="showMarks">
            <span v-for="(box, index) in selected.boxes" :key="index" class="mite-marker"
              :style="position(box, selected)" :aria-label="`瓦螨位置 ${index + 1}`"></span>
          </template>
        </div>
        <dl class="details location-details">
          <div><dt>类别</dt><dd>{{ selected.category === 'mite' ? '有可见瓦螨' : '无可见瓦螨' }}</dd></div>
          <div><dt>可显示的位置</dt><dd>{{ selected.boxes.length }} 处</dd></div>
          <div><dt>原始标签</dt><dd>{{ selected.raw_label }}</dd></div>
          <div><dt>来源</dt><dd>训练集人工标注</dd></div>
        </dl>
        <p v-if="selected.duplicate_annotations" class="caption">这张图有重复的原始标注记录，页面采用首条记录。</p>
        <p v-if="selected.undrawable_boxes" class="caption">{{ selected.undrawable_boxes }} 个原始标注框在图像内没有有效面积。</p>
        <p class="sample-path">{{ selected.name }}</p>
      </template>
      <p v-else class="caption">选择一张训练图片查看。</p>
    </section>
  </div>
</template>
