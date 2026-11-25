<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import ChartPanel from './ChartPanel.vue'
import DetectorPanel from './DetectorPanel.vue'
import LocationsPanel from './LocationsPanel.vue'

const tabs = [['overview', '数据总览'], ['training', '训练任务'], ['comparison', '预处理对照'], ['test', '最终测试'], ['locations', '瓦螨检测']]
const splits = [['train', '训练集'], ['val', '验证集'], ['test', '测试集']]
const methods = { default: '默认裁剪', pad: '等比例补边' }
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
    const [data, history, comparison, baseline, optimized, tuning] = await Promise.all([
      api('/dataset'), api('/runs'), api('/study'), api('/evaluation'),
      api('/evaluation?variant=optimized'), api('/optimization'),
    ])
    dataset.value = data
    runs.value = history
    study.value = comparison
    baselineEvaluation.value = baseline
    optimizedEvaluation.value = optimized
    optimization.value = tuning
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

onMounted(() => {
  refresh()
  timer = setInterval(async () => {
    if (!runs.value.some(run => run.status === 'running') && !starting.value) return
    try {
      runs.value = await api('/runs')
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
    const settings = smoke.value ? { ...form, samples_per_class: 16 } : { ...form }
    const job = await api('/runs', {
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

function openRun(id) {
  activeId.value = id
  tab.value = 'training'
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

const total = computed(() => Object.values(dataset.value?.counts ?? {}).reduce((sum, row) => sum + row.mite + row.no_mite, 0))
const selected = computed(() => study.value?.selection)
const evaluation = computed(() => evaluationVariant.value === 'optimized' ? optimizedEvaluation.value : baselineEvaluation.value)
const testMetrics = computed(() => evaluation.value?.metrics)
const ongoing = computed(() => runs.value.find(run => run.status === 'running'))
const currentEpoch = computed(() => detail.value?.epochs?.length ?? 0)
const completeRuns = computed(() => runs.value.filter(run => run.status === 'completed').length)

const common = { grid: { left: 54, right: 28, top: 48, bottom: 46, containLabel: true }, tooltip: { trigger: 'axis' }, legend: { top: 8 }, animation: false }
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
const losses = computed(() => epochChart([['训练损失', 'train/loss'], ['验证损失', 'val/loss']]))
const accuracies = computed(() => epochChart([['验证准确率', 'metrics/accuracy_top1']], true))

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
      <div class="brand"><span class="brand-mark">V</span><div><strong>瓦螨实验台</strong><small>YOLOv8 分类研究</small></div></div>
      <nav aria-label="功能导航">
        <button v-for="[id, name] in tabs" :key="id" :class="{ current: tab === id }" :aria-current="tab === id ? 'page' : undefined" @click="tab = id">{{ name }}</button>
      </nav>
      <div class="sidebar-status"><span :class="['dot', { connected: online }]"></span>{{ online ? '后端已连接' : loading ? '正在连接' : '等待连接' }}</div>
    </aside>

    <main>
      <header class="topbar">
        <div><h1>{{ tabs.find(([id]) => id === tab)?.[1] }}</h1><p>{{ tab === 'locations' ? '上传蜜蜂近照，查看检测到的瓦螨位置' : '蜜蜂图像中可见瓦螨的二分类实验' }}</p></div>
        <button class="button" @click="refresh">刷新数据</button>
      </header>
      <p v-if="error" class="notice error" role="alert">{{ error }}</p>
      <p v-if="loading && !dataset" class="notice">正在读取数据集和实验记录…</p>

      <template v-if="tab === 'locations'">
        <div class="tools position-tools">
          <button class="button" :class="{ active: positionMode === 'detect' }" :aria-pressed="positionMode === 'detect'" @click="positionMode = 'detect'">上传照片检测</button>
          <button class="button" :class="{ active: positionMode === 'annotations' }" :aria-pressed="positionMode === 'annotations'" @click="positionMode = 'annotations'">查看训练集标注</button>
        </div>
        <DetectorPanel v-if="positionMode === 'detect'" />
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
          <div class="history"><button v-for="run in runs.slice(0, 4)" :key="run.id" @click="openRun(run.id)"><span>{{ run.id }}</span><span>{{ methods[run.settings.preprocessing] }}</span><span>{{ statusText[run.status] }}</span></button></div>
        </section>
      </template>

      <template v-if="tab === 'training'">
        <div class="grid two">
          <section class="panel">
            <header class="panel-head"><h2>启动训练</h2></header>
            <form class="form" @submit.prevent="launch">
              <label>模型<select v-model="form.model" aria-label="模型"><option value="yolov8n-cls">YOLOv8n-cls</option><option value="yolov8s-cls">YOLOv8s-cls</option></select></label>
              <label>预处理<select v-model="form.preprocessing"><option value="default">默认裁剪</option><option value="pad">等比例补边</option></select></label>
              <label>训练轮数<input v-model.number="form.epochs" type="number" min="1" max="200" required></label>
              <label>输入尺寸<input v-model.number="form.imgsz" type="number" min="32" max="640" step="32" required></label>
              <label>批大小<input v-model.number="form.batch" type="number" min="1" max="64" required></label>
              <label>随机种子<input v-model.number="form.seed" type="number" min="0" max="4294967295" required></label>
              <label class="check"><input v-model="smoke" type="checkbox">小样本试跑（每类 16 张）</label>
              <button class="button primary" :disabled="starting || !!ongoing || !online">{{ ongoing ? '训练进行中' : starting ? '启动中…' : '开始训练' }}</button>
            </form>
          </section>
          <section class="panel">
            <header class="panel-head"><h2>实验记录</h2><span v-if="detail" :class="['status', detail.status]">{{ statusText[detail.status] }}</span></header>
            <label class="run-select">选择实验<select v-model="activeId"><option disabled value="">请选择实验</option><option v-for="run in runs" :key="run.id" :value="run.id">{{ run.id }} · {{ statusText[run.status] }}</option></select></label>
            <template v-if="detail">
              <dl class="details"><div><dt>模型</dt><dd>{{ detail.settings.model ?? 'yolov8n-cls' }}</dd></div><div><dt>预处理</dt><dd>{{ methods[detail.settings.preprocessing] }}</dd></div><div><dt>进度</dt><dd>{{ currentEpoch }} / {{ detail.settings.epochs }} 轮</dd></div><div><dt>验证召回率</dt><dd>{{ percent(detail.validation?.mite_recall) }}</dd></div><div><dt>随机种子</dt><dd>{{ detail.settings.seed }}</dd></div></dl>
              <progress :max="detail.settings.epochs" :value="currentEpoch" aria-label="训练进度"></progress>
              <div class="tools files"><a v-for="file in detail.files" :key="file" class="button small" :href="`/api/runs/${detail.id}/files/${file}`">{{ file.split('/').pop() }}</a></div>
            </template>
          </section>
        </div>
        <div v-if="detail" class="grid two">
          <ChartPanel title="训练与验证损失" filename="training-loss" :options="losses" />
          <ChartPanel title="每轮验证准确率" filename="validation-accuracy" :options="accuracies" />
        </div>
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
