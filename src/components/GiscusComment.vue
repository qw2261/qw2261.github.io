<template>
  <div ref="giscusContainer" class="mt-10 pt-8 border-t border-gray-100"></div>
</template>

<script setup>
import { ref, onMounted, watch, nextTick } from 'vue'

const props = defineProps({
  term: {
    type: String,
    required: true
  }
})

const giscusContainer = ref(null)

const GISCUS_CONFIG = {
  repo: 'qw2261/qw2261.github.io',
  repoId: 'R_kgDOSTk40g',
  category: 'Blog Comments',
  categoryId: 'DIC_kwDOSTk40s4C8UY2',
  mapping: 'url',
  reactionsEnabled: '1',
  emitMetadata: '0',
  inputPosition: 'bottom',
  theme: 'light',
  lang: 'zh-CN'
}

function renderGiscus() {
  if (!giscusContainer.value) return
  giscusContainer.value.innerHTML = ''

  const scriptEl = document.createElement('script')
  scriptEl.src = 'https://giscus.app/client.js'
  scriptEl.setAttribute('data-repo', GISCUS_CONFIG.repo)
  scriptEl.setAttribute('data-repo-id', GISCUS_CONFIG.repoId)
  scriptEl.setAttribute('data-category', GISCUS_CONFIG.category)
  scriptEl.setAttribute('data-category-id', GISCUS_CONFIG.categoryId)
  scriptEl.setAttribute('data-mapping', GISCUS_CONFIG.mapping)
  scriptEl.setAttribute('data-term', props.term)
  scriptEl.setAttribute('data-reactions-enabled', GISCUS_CONFIG.reactionsEnabled)
  scriptEl.setAttribute('data-emit-metadata', GISCUS_CONFIG.emitMetadata)
  scriptEl.setAttribute('data-input-position', GISCUS_CONFIG.inputPosition)
  scriptEl.setAttribute('data-theme', GISCUS_CONFIG.theme)
  scriptEl.setAttribute('data-lang', GISCUS_CONFIG.lang)
  scriptEl.setAttribute('crossorigin', 'anonymous')
  scriptEl.async = true

  giscusContainer.value.appendChild(scriptEl)
}

onMounted(() => renderGiscus())

watch(() => props.term, () => {
  nextTick(() => renderGiscus())
})
</script>
