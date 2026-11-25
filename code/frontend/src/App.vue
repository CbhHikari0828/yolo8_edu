<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import ChartPanel from './ChartPanel.vue'
import ImageConfusionMatrix from './ImageConfusionMatrix.vue'
import DetectorPanel from './DetectorPanel.vue'
import LocationsPanel from './LocationsPanel.vue'

const tabs = [['overview', '数据总览'], ['training', '训练任务'], ['comparison', '预处理对照'], ['algorithms', '算法对照'], ['test', '最终测试'], ['locations', '瓦螨检测']]
const splits = [['train', '训练集'], ['val', '验证集'], ['test', '测试集']]
const methods = { default: '默认裁剪', pad: '等比例补边' }
const tasks = { classify: '有无分类训练', detect: '瓦螨定位训练' }
const labels = { mite: '有可见瓦螨', no_mite: '无可见瓦螨' }
const statusText = { running: '训练中', completed: '已完成', failed: '失败', interrupted: '已中断' }
const tab = ref('overview')
const positionMode = ref('detect')
const dataset = ref(null)
const runs = ref([])
const study = ref(null)
const baselineEvaluation = ref(null)
const optimizedEvaluation = ref(null)
const optimization = ref(null)
const classifierComparison = ref(null)
const detectorComparison = ref([])
const comparisonMode = ref('classification')
const detectorGroup = ref(0)
const evaluationVariant = ref(null)
const activeId = ref('')
const detail = ref(null)
const error = ref('')
const loading = ref(true)
const online = ref(false)
const starting = ref(false)
const gallery = ref('misses')
const opened = ref(null)
const viewer = ref(null)
const smoke = ref(false)
const form = reactive({ model: 'yolov8n-cls', preprocessing: 'default', epochs: 10, imgsz: 160, batch: 32, seed: 42 })
const trainingMode = ref('classify')
const detectionForm = reactive({ model: 'yolov8n', initialization: 'pretrained', benchmark: false, epochs: 18, imgsz: 384, batch: 16, seed: 2026 })
const detectorModels = { 'yolov3-tiny': 'YOLOv3-tiny', yolov5n: 'YOLOv5n', 'yolov7-tiny': 'YOLOv7-tiny', yolov8n: 'YOLOv8n' }
const trainingParams = computed(() => trainingMode.value === 'detect' ? detectionForm : form)
let timer

function percent(value) {
  return typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : '—'
}

function number(value) {
  return typeof value === 'number' ? value.toLocaleString('zh-CN') : '—'
}

async function api(path, options) {
  const response = await fetch(`/api${path}`, options)
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(typeof body?.detail === 'string' ? body.detail : `请求失败（${response.status}）`)
  }
  return response.json()
}

async function loadDetail(id) {
  if (!id) return
  const record = await api(`/runs/${id}`)
  if (activeId.value === id) detail.value = record
}

async function refresh() {
  try {
    const [data, history, comparison, baseline, optimized, tuning, classifiers, detectors] = await Promise.all([
      api('/dataset'), api('/runs'), api('/study'), api('/evaluation'),
      api('/evaluation?variant=optimized'), api('/optimization'), api('/classifier-comparison'), api('/detector-comparison'),
    ])
    dataset.value = data
    runs.value = history
    study.value = comparison
    baselineEvaluation.value = baseline
    optimizedEvaluation.value = optimized
    optimization.value = tuning
    classifierComparison.value = classifiers
    detectorComparison.value = detectors
    if (detectorGroup.value >= detectors.length) detectorGroup.value = 0
    evaluationVariant.value ??= optimized ? 'optimized' : 'baseline'
    activeId.value ||= study.value?.selection.run_id ?? history[0]?.id ?? ''
    await loadDetail(activeId.value)
    online.value = true
    error.value = ''
  } catch (failure) {
    error.value = failure.message
    online.value = false
  } finally {
    loading.value = false
  }
}

watch(activeId, async id => {
  if (detail.value?.id === id) return
  detail.value = null
  try { await loadDetail(id) } catch (failure) { error.value = failure.message }
})

watch(() => [detectionForm.model, detectionForm.benchmark], ([model, benchmark]) => {
  if (model !== 'yolov8n' || benchmark) detectionForm.initialization = 'pretrained'
})

onMounted(() => {
  refresh()
  timer = setInterval(async () => {
    if (!runs.value.some(run => run.status === 'running') && !starting.value) return
    try {
      runs.value = await api('/runs')
      if (!runs.value.some(run => run.status === 'running')) detectorComparison.value = await api('/detector-comparison')
      await loadDetail(activeId.value)
    } catch (failure) {
      error.value = failure.message
    }
  }, 3000)
})
onBeforeUnmount(() => clearInterval(timer))

async function launch() {
  starting.value = true
  error.value = ''
  try {
    const isDetection = trainingMode.value === 'detect'
    const settings = { ...trainingParams.value }
    if (!isDetection && smoke.value) settings.samples_per_class = 16
    const job = await api(isDetection ? '/runs/detection' : '/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    })
    activeId.value = job.id
    runs.value = [job, ...runs.value]
    await loadDetail(job.id)
  } catch (failure) {
    error.value = failure.message
  } finally {
    starting.value = false
  }
}

async function resumeRun(id) {
  starting.value = true
  error.value = ''
  try {
    const job = await api(`/runs/${id}/resume`, { method: 'POST' })
    runs.value = runs.value.map(run => run.id === id ? job : run)
    await loadDetail(id)
  } catch (failure) {
    error.value = failure.message
  } finally {
    starting.value = false
  }
}

function openRun(id) {
  activeId.value = id
  trainingMode.value = id.startsWith('mite-detection') ? 'detect' : 'classify'
  tab.value = 'training'
}

function openDetector() {
  positionMode.value = 'detect'
  tab.value = 'locations'
}

async function openImage(sample) {
  opened.value = sample
  await nextTick()
  viewer.value?.showModal()
}

function saveTable(name, rows) {
  const text = rows.map(row => row.map(value => '"' + String(value).replaceAll('"', '""') + '"').join(',')).join('\r\n')
  const url = URL.createObjectURL(new Blob(['\ufeff', text], { type: 'text/csv;charset=utf-8' }))
  const link = document.createElement('a')
  link.href = url
  link.download = `${name}.csv`
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

const classifierMetrics = [
  ['准确率', 'accuracy'], ['宏平均 F1', 'macro_f1'], ['有螨召回率', 'mite_recall'],
  ['有螨漏判', 'mite_false_negatives'], ['无螨误报', 'mite_false_positives'],
]

function modelName(model) {
  return model === 'resnet18' ? 'ResNet18' : detectorModels[model] ?? model
}

function exportClassifierComparison() {
  saveTable('classifier-comparison', [
    ['模型', '实验编号', ...classifierMetrics.map(([name]) => name)],
    ...classifierComparison.value.models.map(({ metrics }) => [
      modelName(metrics.model), metrics.source_run, ...classifierMetrics.map(([, key]) => metrics[key]),
    ]),
  ])
}

const total = computed(() => Object.values(dataset.value?.counts ?? {}).reduce((sum, row) => sum + row.mite + row.no_mite, 0))
const selected = computed(() => study.value?.selection)
const evaluation = computed(() => evaluationVariant.value === 'optimized' ? optimizedEvaluation.value : baselineEvaluation.value)
const testMetrics = computed(() => evaluation.value?.metrics)
const ongoing = computed(() => runs.value.find(run => run.status === 'running'))
const currentEpoch = computed(() => detail.value?.epochs?.length ?? 0)
const detectionDetail = computed(() => detail.value?.task === 'detect')
const evaluatingDetection = computed(() => detectionDetail.value && detail.value?.status === 'running' && (
  currentEpoch.value >= detail.value.settings.epochs ||
  detail.value.log?.includes('Selecting confidence from validation images.')
))
const completeRuns = computed(() => runs.value.filter(run => run.status === 'completed').length)

const common = { grid: { left: 54, right: 28, top: 48, bottom: 46, containLabel: true }, tooltip: { trigger: 'axis' }, legend: { top: 8 }, animation: false }
const currentDetectorGroup = computed(() => detectorComparison.value[detectorGroup.value])
const detectorRows = computed(() => currentDetectorGroup.value?.models ?? [])
const detectorMatrixMax = computed(() => Math.max(1, ...detectorRows.value.flatMap(run => run.test.image.confusion_matrix.flat())))

function exportDetectorComparison() {
  saveTable('detector-comparison', [
    ['模型', '实验编号', 'mAP@0.50', 'mAP@0.50:0.95', '位置框 F1', '单图召回率', '漏检照片', '误报照片'],
    ...detectorRows.value.map(run => [
      modelName(run.settings.model), run.id, run.test.map50, run.test.map50_95, run.test.boxes_at_iou50.f1,
      run.test.image.mite_recall, run.test.image.mite_false_negatives, run.test.image.mite_false_positives,
    ]),
  ])
}

const detectorChart = computed(() => ({
  ...common,
  xAxis: { type: 'category', data: ['mAP@0.50', 'mAP@0.50:0.95', '位置框 F1'] },
  yAxis: { type: 'value', min: 0, max: 100, axisLabel: { formatter: '{value}%' } },
  series: detectorRows.value.map(run => ({
    name: modelName(run.settings.model), type: 'bar', barMaxWidth: 50,
    data: [run.test.map50, run.test.map50_95, run.test.boxes_at_iou50.f1].map(value => Number((100 * value).toFixed(2))),
  })),
}))

function detectorEpochChart(key, asPercentage = false) {
  return {
    ...common,
    xAxis: { type: 'value', name: '轮次', min: 1, minInterval: 1 },
    yAxis: { type: 'value', min: 0, max: asPercentage ? 100 : undefined, axisLabel: { formatter: asPercentage ? '{value}%' : '{value}' } },
    series: detectorRows.value.map(run => ({
      name: modelName(run.settings.model), type: 'line', showSymbol: run.epochs.length < 3,
      data: run.epochs.filter(epoch => epoch[key] != null).map(epoch => [epoch.epoch, Number((epoch[key] * (asPercentage ? 100 : 1)).toFixed(5))]),
    })),
  }
}

const prChart = computed(() => ({
  ...common,
  xAxis: { type: 'value', name: '召回率', min: 0, max: 1, nameLocation: 'middle', nameGap: 27 },
  yAxis: { type: 'value', name: '精确率', min: 0, max: 1, nameLocation: 'middle', nameGap: 44 },
  series: detectorRows.value.filter(run => run.test.pr50).map(run => ({
    name: modelName(run.settings.model), type: 'line', showSymbol: run.test.pr50.length < 3, data: run.test.pr50,
  })),
}))
const classifierChart = computed(() => ({
  ...common,
  xAxis: { type: 'category', data: classifierMetrics.slice(0, 3).map(([name]) => name) },
  yAxis: { type: 'value', min: 0, max: 100, axisLabel: { formatter: '{value}%' } },
  series: (classifierComparison.value?.models ?? []).map(({ metrics }) => ({
    name: modelName(metrics.model), type: 'bar', barMaxWidth: 52, label: { show: true, position: 'top', formatter: '{c}%' },
    data: classifierMetrics.slice(0, 3).map(([, key]) => Number((100 * metrics[key]).toFixed(2))),
  })),
}))
const datasetChart = computed(() => ({
  ...common,
  xAxis: { type: 'category', data: splits.map(([, label]) => label) },
  yAxis: { type: 'value' },
  series: ['mite', 'no_mite'].map(key => ({
    name: labels[key], type: 'bar', stack: 'images', barMaxWidth: 66,
    data: splits.map(([split]) => dataset.value?.counts?.[split]?.[key] ?? 0),
  })),
}))

function epochChart(series, isPercent = false) {
  return {
    ...common,
    xAxis: { type: 'category', name: '轮次', data: detail.value?.epochs?.map(row => row.epoch) ?? [] },
    yAxis: { type: 'value', min: isPercent ? 0 : undefined, max: isPercent ? 100 : undefined, axisLabel: { formatter: isPercent ? '{value}%' : '{value}' } },
    series: series.map(([name, key]) => ({
      name, type: 'line', symbolSize: 7,
      data: detail.value?.epochs?.map(row => row[key] == null ? null : Number((row[key] * (isPercent ? 100 : 1)).toFixed(4))) ?? [],
    })),
  }
}
const losses = computed(() => epochChart(detectionDetail.value
  ? [['训练框损失', 'train/box_loss'], ['验证框损失', 'val/box_loss']]
  : [['训练损失', 'train/loss'], ['验证损失', 'val/loss']]))
const accuracies = computed(() => epochChart(detectionDetail.value
  ? [['mAP@0.50', 'metrics/mAP50(B)'], ['mAP@0.50:0.95', 'metrics/mAP50-95(B)']]
  : [['验证准确率', 'metrics/accuracy_top1']], true))

const comparisonGroups = computed(() => {
  if (study.value) return study.value.groups
  const defaultRun = runs.value.find(run => run.id === 'compare-default')
  const padRun = runs.value.find(run => run.id === 'compare-pad')
  if (!defaultRun?.validation || !padRun?.validation) return null
  return {
    default: { runs: [{ ...defaultRun.validation, run_id: defaultRun.id, seed: defaultRun.settings.seed }], mean: defaultRun.validation },
    pad: { runs: [{ ...padRun.validation, run_id: padRun.id, seed: padRun.settings.seed }], mean: padRun.validation },
  }
})
const compareFields = [
  ['有螨召回率', 'mite_recall'], ['宏平均 F1', 'macro_f1'], ['验证准确率', 'accuracy'],
  ['有螨漏判率', 'mite_false_negative_rate'], ['有螨漏判', 'mite_false_negatives'], ['无螨误报', 'mite_false_positives'],
]
const comparisonChart = computed(() => ({
  ...common,
  xAxis: { type: 'category', data: compareFields.slice(0, 3).map(([name]) => name) },
  yAxis: { type: 'value', min: 0, max: 100, axisLabel: { formatter: '{value}%' } },
  series: Object.entries(comparisonGroups.value ?? {}).map(([mode, group]) => ({
    name: methods[mode], type: 'bar', barMaxWidth: 50, label: { show: true, position: 'top', formatter: '{c}%' },
    data: compareFields.slice(0, 3).map(([, key]) => Number((100 * group.mean[key]).toFixed(2))),
  })),
}))

function formatCompare(value, key) {
  return ['mite_false_negatives', 'mite_false_positives'].includes(key)
    ? (typeof value === 'number' ? value.toLocaleString('zh-CN', { maximumFractionDigits: 2 }) : '—')
    : percent(value)
}

function exportComparison() {
  const rows = Object.entries(comparisonGroups.value).flatMap(([mode, group]) =>
    group.runs.map(run => [methods[mode], run.seed, run.run_id, ...compareFields.map(([, key]) => run[key])]))
  saveTable('validation-comparison', [['方案', '种子', '实验', ...compareFields.map(([name]) => name)], ...rows])
}

const heatmap = computed(() => {
  const matrix = testMetrics.value?.confusion_matrix ?? [[0, 0], [0, 0]]
  const names = (testMetrics.value?.class_names ?? ['mite', 'no_mite']).map(name => labels[name])
  return {
    animation: false,
    grid: { left: 38, right: 24, top: 36, bottom: 56, containLabel: true },
    tooltip: { formatter: item => `真实：${names[item.value[1]]}<br/>预测：${names[item.value[0]]}<br/>${number(item.value[2])} 张` },
    xAxis: { type: 'category', data: names, name: '预测类别', nameLocation: 'middle', nameGap: 34 },
    yAxis: { type: 'category', data: names, name: '真实类别', inverse: true },
    visualMap: { show: false, min: 0, max: Math.max(...matrix.flat(), 1), inRange: { color: ['#f3f7f5', '#8ce2ce'] } },
    series: [{ type: 'heatmap', data: matrix.flatMap((row, y) => row.map((value, x) => [x, y, value])), label: { show: true, color: '#143a34', fontWeight: 700, fontSize: 20 }, itemStyle: { borderColor: '#ffffff', borderWidth: 4 } }],
  }
})
const testFields = [
  ['准确率', 'accuracy'], ['宏平均 F1', 'macro_f1'], ['有螨精确率', 'mite_precision'],
  ['有螨召回率', 'mite_recall'], ['有螨 F1', 'mite_f1'], ['漏判率', 'mite_false_negative_rate'],
]
const testComparisonFields = [
  ['准确率', 'accuracy'], ['宏平均 F1', 'macro_f1'], ['有螨召回率', 'mite_recall'],
  ['有螨漏判', 'mite_false_negatives'], ['无螨误报', 'mite_false_positives'],
]

function evaluationLink(path) {
  return `/api/evaluation/${path}?variant=${evaluationVariant.value}`
}
</script>

<template>
  <div class="workspace">
    <aside class="sidebar">
      <div class="brand"><span class="brand-mark">V</span><div><strong>瓦螨实验台</strong><small>YOLOv8 训练与检测</small></div></div>
      <nav aria-label="功能导航">
        <button v-for="[id, name] in tabs" :key="id" :class="{ current: tab === id }" :aria-current="tab === id ? 'page' : undefined" @click="tab = id">{{ name }}</button>
      </nav>
      <div class="sidebar-status"><span :class="['dot', { connected: online }]"></span>{{ online ? '后端已连接' : loading ? '正在连接' : '等待连接' }}</div>
    </aside>

    <main>
      <header class="topbar">
        <div><h1>{{ tabs.find(([id]) => id === tab)?.[1] }}</h1><p>{{ tab === 'locations' ? '上传蜜蜂近照，查看检测到的瓦螨位置' : '蜜蜂图像中的瓦螨分类与定位实验' }}</p></div>
        <button class="button" @click="refresh">刷新数据</button>
      </header>
      <p v-if="error" class="notice error" role="alert">{{ error }}</p>
      <p v-if="loading && !dataset" class="notice">正在读取数据集和实验记录…</p>

      <template v-if="tab === 'locations'">
        <div class="tools position-tools">
          <button class="button" :class="{ active: positionMode === 'detect' }" :aria-pressed="positionMode === 'detect'" @click="positionMode = 'detect'">上传照片检测</button>
          <button class="button" :class="{ active: positionMode === 'annotations' }" :aria-pressed="positionMode === 'annotations'" @click="positionMode = 'annotations'">查看训练集标注</button>
        </div>
        <DetectorPanel v-if="positionMode === 'detect'" @open-run="openRun" />
        <LocationsPanel v-else />
      </template>

      <template v-if="tab === 'overview'">
        <div class="credentials">
          <section class="stat"><span>数据集图像</span><strong>{{ dataset ? number(total) : '—' }}</strong><small>训练、验证与测试</small></section>
          <section class="stat"><span>已完成训练</span><strong>{{ number(completeRuns) }}</strong><small>{{ ongoing ? '当前有任务运行中' : '可回溯实验记录' }}</small></section>
          <section class="stat"><span>对照组数</span><strong>{{ number(comparisonGroups?.default.runs.length ?? 0) }}</strong><small>匹配随机种子的两种预处理</small></section>
          <section class="stat accent"><span>最终测试准确率</span><strong>{{ percent(testMetrics?.accuracy) }}</strong><small>{{ testMetrics ? `${number(testMetrics.samples)} 张独立测试图` : '等待最终评估' }}</small></section>
        </div>
        <div v-if="dataset" class="grid two">
          <ChartPanel title="数据集类别分布" filename="varroa-dataset" :options="datasetChart" />
          <section class="panel">
            <header class="panel-head"><h2>样本统计</h2><button class="button small" @click="saveTable('dataset-counts', [['划分', '有螨', '无螨'], ...splits.map(([key, name]) => [name, dataset.counts[key].mite, dataset.counts[key].no_mite])])">导出 CSV</button></header>
            <table><thead><tr><th>数据划分</th><th>有螨</th><th>无螨</th><th>总计</th></tr></thead>
              <tbody><tr v-for="[key, name] in splits" :key="key"><th>{{ name }}</th><td>{{ number(dataset.counts[key].mite) }}</td><td>{{ number(dataset.counts[key].no_mite) }}</td><td>{{ number(dataset.counts[key].mite + dataset.counts[key].no_mite) }}</td></tr></tbody>
            </table>
            <div class="caption">原始标签 0 为无可见瓦螨，1 和 3 合并为有可见瓦螨。</div>
          </section>
        </div>
        <section v-if="runs.length" class="panel">
          <header class="panel-head"><h2>最近的实验</h2><button class="button small" @click="tab = 'training'">查看训练任务</button></header>
          <div class="history"><button v-for="run in runs.slice(0, 4)" :key="run.id" @click="openRun(run.id)"><span>{{ run.id }}</span><span>{{ run.task === 'detect' ? '瓦螨定位' : methods[run.settings.preprocessing] }}</span><span>{{ statusText[run.status] }}</span></button></div>
        </section>
      </template>

      <template v-if="tab === 'training'">
        <div class="grid two">
          <section class="panel">
            <header class="panel-head"><h2>启动训练</h2></header>
            <div class="tools position-tools" aria-label="训练类型">
              <button v-for="(label, type) in tasks" :key="type" type="button" class="button"
                :class="{ active: trainingMode === type }" :aria-pressed="trainingMode === type"
                @click="trainingMode = type">{{ label }}</button>
            </div>
            <form class="form" @submit.prevent="launch">
              <template v-if="trainingMode === 'classify'">
                <label>模型<select v-model="form.model" aria-label="模型"><option value="yolov8n-cls">YOLOv8n-cls</option><option value="yolov8s-cls">YOLOv8s-cls</option><option value="resnet18">ResNet18</option></select></label>
                <label>预处理<select v-model="form.preprocessing"><option value="default">默认裁剪</option><option value="pad">等比例补边</option></select></label>
              </template>
              <template v-else>
                <label>定位模型<select v-model="detectionForm.model" aria-label="定位模型"><option v-for="(label, model) in detectorModels" :key="model" :value="model">{{ label }}</option></select></label>
                <label>初始权重<select v-model="detectionForm.initialization" aria-label="初始权重"><option value="pretrained">官方预训练权重</option><option value="active" :disabled="detectionForm.model !== 'yolov8n' || detectionForm.benchmark">当前检测模型（继续微调）</option></select></label>
              </template>
              <label>训练轮数<input v-model.number="trainingParams.epochs" type="number" min="1" max="200" required></label>
              <label>输入尺寸<input v-model.number="trainingParams.imgsz" type="number" :min="trainingMode === 'detect' || form.model === 'resnet18' ? 64 : 32" max="640" step="32" required></label>
              <label>批大小<input v-model.number="trainingParams.batch" type="number" min="1" max="64" required></label>
              <label>随机种子<input v-model.number="trainingParams.seed" type="number" min="0" max="4294967295" required></label>
              <label v-if="trainingMode === 'classify'" class="check"><input v-model="smoke" type="checkbox">小样本试跑（每类 16 张）</label>
              <template v-else>
                <label v-if="detectionForm.model === 'yolov8n'" class="check"><input v-model="detectionForm.benchmark" type="checkbox">作为定位对照</label>
                <p class="caption form-note">{{ detectionForm.model !== 'yolov8n' || detectionForm.benchmark ? '任务完成后将生成定位对照指标和混淆矩阵。' : '评估完成后，新权重自动用于「瓦螨检测」。' }}</p>
              </template>
              <button class="button primary" :disabled="starting || !!ongoing || !online">{{ ongoing ? '训练进行中' : starting ? '启动中…' : trainingMode === 'detect' ? '开始定位训练' : '开始训练' }}</button>
            </form>
          </section>
          <section class="panel">
            <header class="panel-head"><h2>实验记录</h2><span v-if="detail" :class="['status', detail.status]">{{ evaluatingDetection ? '评估中' : statusText[detail.status] }}</span></header>
            <label class="run-select">选择实验<select v-model="activeId"><option disabled value="">请选择实验</option><option v-for="run in runs" :key="run.id" :value="run.id">{{ tasks[run.task] }} · {{ run.id }} · {{ statusText[run.status] }}</option></select></label>
            <template v-if="detail">
              <dl class="details">
                <div><dt>模型</dt><dd>{{ detail.settings.model ?? 'yolov8n-cls' }}</dd></div>
                <div v-if="detectionDetail"><dt>训练入口</dt><dd>{{ detail.origin === 'web' ? '网页训练任务' : '后端命令' }}</dd></div>
                <div v-else><dt>预处理</dt><dd>{{ methods[detail.settings.preprocessing] }}</dd></div>
                <div><dt>进度</dt><dd>{{ currentEpoch }} / {{ detail.settings.epochs ?? '—' }} 轮</dd></div>
                <div><dt>验证集有螨召回率</dt><dd>{{ percent(detectionDetail ? detail.validation?.image.mite_recall : detail.validation?.mite_recall) }}</dd></div>
                <div><dt>随机种子</dt><dd>{{ detail.settings.seed ?? '—' }}</dd></div>
                <div v-if="detectionDetail"><dt>初始权重</dt><dd>{{ detail.settings.initial_source_run ?? (detail.settings.initialization === 'pretrained' ? '官方预训练权重' : '—') }}</dd></div>
              </dl>
              <progress v-if="detail.settings.epochs" :max="detail.settings.epochs" :value="currentEpoch" aria-label="训练进度"></progress>
              <p v-if="detail.status === 'interrupted'" class="caption">后端停止或重启，任务未完成。</p>
              <button v-if="detail.can_resume" class="button small" :disabled="starting || !!ongoing" @click="resumeRun(detail.id)">从检查点续训</button>
              <button v-if="detail.serving" class="button small" @click="openDetector">当前用于瓦螨检测</button>
              <div class="tools files"><a v-for="file in detail.files" :key="file" class="button small" :href="`/api/runs/${detail.id}/files/${file}`">{{ file.split('/').pop() }}</a></div>
            </template>
          </section>
        </div>
        <div v-if="detail" class="grid two">
          <ChartPanel :title="detectionDetail ? '训练与验证定位损失' : '训练与验证损失'" filename="training-loss" :options="losses" />
          <ChartPanel :title="detectionDetail ? '每轮验证集位置检测 mAP' : '每轮验证准确率'" filename="validation-metrics" :options="accuracies" />
        </div>
        <section v-if="detectionDetail && detail?.test" class="panel">
          <header class="panel-head"><h2>位置检测测试结果</h2><span>{{ number(detail.test.image.samples) }} 张测试图</span></header>
          <table><tbody>
            <tr><th>位置检测 mAP@0.50</th><td>{{ percent(detail.test.map50) }}</td></tr>
            <tr><th>位置检测 mAP@0.50:0.95</th><td>{{ percent(detail.test.map50_95) }}</td></tr>
            <tr><th>位置框 F1（IoU ≥ 0.50）</th><td>{{ percent(detail.test.boxes_at_iou50.f1) }}</td></tr>
            <tr><th>单图有无判断准确率</th><td>{{ percent(detail.test.image.accuracy) }}</td></tr>
            <tr><th>有螨图片召回率</th><td>{{ percent(detail.test.image.mite_recall) }}</td></tr>
            <tr><th>验证集选定的置信度阈值</th><td>{{ detail.confidence_threshold.toFixed(2) }}</td></tr>
          </tbody></table>
        </section>
        <ImageConfusionMatrix v-if="detectionDetail && detail?.test" title="测试集有无瓦螨混淆矩阵"
          :filename="`${detail.id}-confusion-matrix`" :matrix="detail.test.image.confusion_matrix" />
        <section v-if="detail?.log" class="panel"><header class="panel-head"><h2>运行日志</h2></header><pre>{{ detail.log }}</pre></section>
      </template>

      <template v-if="tab === 'comparison'">
        <p v-if="!comparisonGroups && !loading" class="notice">等待两种预处理的匹配实验完成。</p>
        <template v-if="comparisonGroups">
          <div v-if="selected" class="selection"><strong>验证集选定方案 · {{ methods[selected.preprocessing] }}</strong><span>采用 {{ selected.run_id }} 的权重</span><a href="/api/study/file" class="button small">下载对照 JSON</a></div>
          <ChartPanel title="两种预处理的验证指标" filename="preprocessing-comparison" :options="comparisonChart" />
          <section class="panel">
            <header class="panel-head"><h2>配对实验统计</h2><div class="tools"><span class="muted">验证集 · {{ comparisonGroups.default.runs.length }} 个种子</span><button class="button small" @click="exportComparison">导出 CSV</button></div></header>
            <table><thead><tr><th>指标</th><th>默认裁剪</th><th>等比例补边</th></tr></thead>
              <tbody><tr v-for="[name, key] in compareFields" :key="key"><th>{{ name }}</th><td v-for="mode in ['default', 'pad']" :key="mode">{{ formatCompare(comparisonGroups[mode].mean[key], key) }}<small v-if="comparisonGroups[mode].std?.[key] != null"> ± {{ formatCompare(comparisonGroups[mode].std[key], key) }}</small></td></tr></tbody>
            </table>
          </section>
          <section class="panel">
            <header class="panel-head"><h2>各次实验</h2></header>
            <table><thead><tr><th>实验</th><th>种子</th><th>召回率</th><th>漏判</th><th>宏平均 F1</th></tr></thead>
              <tbody><tr v-for="run in [...comparisonGroups.default.runs, ...comparisonGroups.pad.runs]" :key="run.run_id"><th><button class="link-button" @click="openRun(run.run_id)">{{ run.run_id }}</button></th><td>{{ run.seed }}</td><td>{{ percent(run.mite_recall) }}</td><td>{{ run.mite_false_negatives }}</td><td>{{ percent(run.macro_f1) }}</td></tr></tbody>
            </table>
          </section>
        </template>
      </template>

      <template v-if="tab === 'algorithms'">
        <div class="tools position-tools">
          <button class="button" :class="{ active: comparisonMode === 'classification' }" @click="comparisonMode = 'classification'">分类算法</button>
          <button class="button" :class="{ active: comparisonMode === 'detection' }" @click="comparisonMode = 'detection'">定位模型</button>
        </div>
        <template v-if="comparisonMode === 'classification' && classifierComparison">
          <section class="panel">
            <header class="panel-head"><h2>分类算法测试对照</h2><button class="button small" @click="exportClassifierComparison">导出 CSV</button></header>
            <p class="caption">官方测试集 {{ number(classifierComparison.samples) }} 张 · 输入 {{ classifierComparison.protocol.imgsz }} · {{ classifierComparison.protocol.epochs }} 轮 · 批大小 {{ classifierComparison.protocol.batch }} · 种子 {{ classifierComparison.protocol.seed }} · 阈值 {{ classifierComparison.threshold.toFixed(2) }}</p>
            <table>
              <thead><tr><th>模型</th><th v-for="[label] in classifierMetrics" :key="label">{{ label }}</th><th>实验</th></tr></thead>
              <tbody><tr v-for="{ metrics } in classifierComparison.models" :key="metrics.source_run">
                <th>{{ modelName(metrics.model) }}</th>
                <td v-for="[, key] in classifierMetrics" :key="key">{{ formatCompare(metrics[key], key) }}</td>
                <td><button class="button small" @click="openRun(metrics.source_run)">{{ metrics.source_run }}</button></td>
              </tr></tbody>
            </table>
          </section>
          <ChartPanel title="分类测试指标" filename="classifier-comparison" :options="classifierChart" />
          <div class="grid two">
            <ImageConfusionMatrix v-for="{ metrics } in classifierComparison.models" :key="metrics.source_run"
              :title="`${modelName(metrics.model)} 混淆矩阵`" :filename="`${metrics.model}-confusion-matrix`" :matrix="metrics.confusion_matrix" />
          </div>
        </template>
        <template v-else-if="comparisonMode === 'detection' && currentDetectorGroup">
          <section class="panel">
            <header class="panel-head"><h2>定位模型测试对照</h2><button class="button small" @click="exportDetectorComparison">导出 CSV</button></header>
            <label class="run-select">对照配置<select v-model="detectorGroup">
              <option v-for="(group, index) in detectorComparison" :key="index" :value="index">
                {{ group.protocol.epochs }} 轮 · 输入 {{ group.protocol.imgsz }} · 批大小 {{ group.protocol.batch }} · 种子 {{ group.protocol.seed }} · {{ group.models.length }} 个模型
              </option>
            </select></label>
            <table><thead><tr><th>模型</th><th>mAP@0.50</th><th>mAP@0.50:0.95</th><th>位置框 F1</th><th>漏检照片</th><th>误报照片</th><th>实验</th></tr></thead>
              <tbody><tr v-for="run in detectorRows" :key="run.id">
                <th>{{ modelName(run.settings.model) }}</th><td>{{ percent(run.test.map50) }}</td><td>{{ percent(run.test.map50_95) }}</td>
                <td>{{ percent(run.test.boxes_at_iou50.f1) }}</td><td>{{ number(run.test.image.mite_false_negatives) }}</td><td>{{ number(run.test.image.mite_false_positives) }}</td>
                <td><button class="button small" @click="openRun(run.id)">{{ run.id }}</button></td>
              </tr></tbody>
            </table>
          </section>
          <ChartPanel title="位置检测测试指标" filename="detector-comparison" :options="detectorChart" />
          <ChartPanel v-if="prChart.series.length" title="测试集 P–R 曲线（IoU 0.50）" filename="detector-pr50" :options="prChart" />
          <p class="caption">训练曲线记录各模型训练器的结果。损失值用于观察各自的收敛趋势。</p>
          <div class="grid two">
            <ChartPanel title="训练框损失对照" filename="detector-train-box-loss" :options="detectorEpochChart('train/box_loss')" />
            <ChartPanel title="验证框损失对照" filename="detector-val-box-loss" :options="detectorEpochChart('val/box_loss')" />
            <ChartPanel title="每轮验证 mAP@0.50 对照" filename="detector-val-map50" :options="detectorEpochChart('metrics/mAP50(B)', true)" />
            <ChartPanel title="每轮验证 mAP@0.50:0.95 对照" filename="detector-val-map50-95" :options="detectorEpochChart('metrics/mAP50-95(B)', true)" />
          </div>
          <div class="grid two">
            <ImageConfusionMatrix v-for="run in detectorRows" :key="run.id" :title="`${modelName(run.settings.model)} 单图混淆矩阵`"
              :filename="`${run.settings.model}-${run.id}-confusion-matrix`" :matrix="run.test.image.confusion_matrix" :max="detectorMatrixMax" />
          </div>
        </template>
        <p v-else-if="comparisonMode === 'detection'" class="notice">完成定位对照任务后，测试指标和混淆矩阵会显示在这里。</p>
        <p v-else class="notice">待同一批测试图片上的算法对照评估完成。</p>
      </template>

      <template v-if="tab === 'test'">
        <p v-if="!testMetrics && !loading" class="notice">测试集结果会在最终评估完成后显示。</p>
        <section v-if="optimization" class="panel">
          <header class="panel-head"><h2>优化方案验证</h2><span class="muted">验证集选择 · 召回率至少 {{ percent(optimization.minimum_mite_recall) }}</span></header>
          <table><thead><tr><th>候选模型</th><th>阈值</th><th>验证准确率</th><th>有螨召回率</th></tr></thead>
            <tbody><tr v-for="candidate in optimization.candidates" :key="candidate.run_id"><th><button class="link-button" @click="openRun(candidate.run_id)">{{ candidate.model }} · {{ candidate.imgsz }}px</button><small v-if="candidate.run_id === optimization.selection.run_id">　选定</small></th><td>{{ candidate.selected?.threshold ?? '—' }}</td><td>{{ percent(candidate.selected?.metrics.accuracy) }}</td><td>{{ percent(candidate.selected?.metrics.mite_recall) }}</td></tr></tbody>
          </table>
        </section>
        <section v-if="optimizedEvaluation" class="panel">
          <header class="panel-head"><h2>同一测试集复评</h2><span class="muted">3,408 张照片</span></header>
          <table><thead><tr><th>指标</th><th>原方案</th><th>优化候选</th></tr></thead>
            <tbody><tr v-for="[name, key] in testComparisonFields" :key="key"><th>{{ name }}</th><td>{{ formatCompare(baselineEvaluation?.metrics[key], key) }}</td><td>{{ formatCompare(optimizedEvaluation.metrics[key], key) }}</td></tr></tbody>
          </table>
        </section>
        <template v-if="testMetrics">
          <div class="selection">
            <label v-if="optimizedEvaluation" class="variant-selector">查看<select v-model="evaluationVariant"><option value="baseline">原方案</option><option value="optimized">优化候选</option></select></label>
            <strong>{{ methods[testMetrics.preprocessing] }} · {{ number(testMetrics.samples) }} 张测试图</strong>
            <button class="link-button" @click="openRun(testMetrics.source_run)">{{ testMetrics.source_run }}</button>
            <div class="tools"><a class="button small" :href="evaluationLink('files/test_metrics.json')">指标 JSON</a><a class="button small" :href="evaluationLink('files/test_confusion_matrix.csv')">矩阵 CSV</a><a class="button small" :href="evaluationLink('files/test_predictions.csv')">逐图 CSV</a></div>
          </div>
          <div class="grid two">
            <ChartPanel title="测试集混淆矩阵" :filename="`test-confusion-matrix-${evaluationVariant}`" :options="heatmap" />
            <section class="panel">
              <header class="panel-head"><h2>最终指标</h2><span class="muted">独立测试集</span></header>
              <table><tbody><tr v-for="[name, key] in testFields" :key="key"><th>{{ name }}</th><td>{{ percent(testMetrics[key]) }}</td></tr><tr><th>有螨漏判</th><td>{{ number(testMetrics.mite_false_negatives) }} 张</td></tr><tr><th>无螨误报</th><td>{{ number(testMetrics.mite_false_positives) }} 张</td></tr></tbody></table>
            </section>
          </div>
          <section class="panel">
            <header class="panel-head"><h2>错例原图</h2><div class="tools"><button class="button small" :class="{ active: gallery === 'misses' }" @click="gallery = 'misses'">漏判</button><button class="button small" :class="{ active: gallery === 'false_alarms' }" @click="gallery = 'false_alarms'">误报</button></div></header>
            <div v-if="!evaluation.examples[gallery].length" class="caption">该类别没有错例。</div>
            <div v-else class="gallery"><button v-for="sample in evaluation.examples[gallery]" :key="sample.index" @click="openImage(sample)"><img loading="lazy" :src="evaluationLink(`images/${sample.index}`)" :alt="`真实${labels[sample.actual]}的蜜蜂`"><span>预测置信度 {{ percent(sample.confidence) }}</span></button></div>
          </section>
        </template>
      </template>
    </main>
  </div>
  <dialog ref="viewer">
    <div v-if="opened" class="viewer"><header class="panel-head"><h2>错例详情</h2><button class="button" @click="viewer.close()">关闭</button></header><img :src="evaluationLink(`images/${opened.index}`)" :alt="labels[opened.actual]"><dl class="details"><div><dt>真实标签</dt><dd>{{ labels[opened.actual] }}</dd></div><div><dt>预测标签</dt><dd>{{ labels[opened.predicted] }}</dd></div><div><dt>预测置信度</dt><dd>{{ percent(opened.confidence) }}</dd></div></dl><p class="sample-path">{{ opened.image }}</p></div>
  </dialog>
</template>
