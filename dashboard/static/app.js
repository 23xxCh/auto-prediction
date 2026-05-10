/* ============================================
   智能装备健康管理系统 — 数字孪生看板脚本
   ============================================ */

// ========== 全局配置 ==========
const POLL_INTERVAL = 3000;  // 数据轮询间隔（毫秒）
const API_BASE = '';         // API基础路径，空字符串表示同源

// ECharts图表实例
let rulChart = null;
let gaugeTemp = null;
let gaugeVib = null;
let gaugeCurr = null;
let gaugeRpm = null;

// 当前选中设备
let selectedDeviceId = null;

// ========== Toast 通知 ==========

/**
 * 显示 Toast 通知
 * @param {string} message - 通知内容
 * @param {string} type - 类型 ('error' | 'success')
 * @param {number} duration - 显示时长（毫秒），默认 4000
 */
function showToast(message, type = 'error', duration = 4000) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<p class="toast-message">${message}</p>`;
  container.appendChild(toast);

  // 自动移除
  setTimeout(() => {
    removeToast(toast);
  }, duration);
}

/**
 * 移除 Toast 通知（带动画）
 * @param {HTMLElement} toast - Toast 元素
 */
function removeToast(toast) {
  toast.style.animation = 'toastOut 0.3s ease forwards';
  toast.addEventListener('animationend', () => {
    if (toast.parentNode) {
      toast.parentNode.removeChild(toast);
    }
  });
}

// ========== 工具函数 ==========

/**
 * 格式化时间显示
 * @param {Date} date - 日期对象
 * @returns {string} 格式化的时间字符串
 */
function formatTime(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  const seconds = String(date.getSeconds()).padStart(2, '0');
  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}

/**
 * 获取状态对应的CSS类名
 * @param {string} status - 设备状态
 * @returns {string} CSS类名
 */
function getStatusClass(status) {
  switch (status) {
    case 'normal': return 'status-normal';
    case 'warning': return 'status-warning';
    case 'danger': return 'status-danger';
    default: return 'status-normal';
  }
}

/**
 * 获取RUL等级CSS类名
 * @param {number} rul - 剩余使用寿命
 * @returns {string} CSS类名
 */
function getRULClass(rul) {
  if (rul >= 60) return 'rul-high';
  if (rul >= 30) return 'rul-medium';
  return 'rul-low';
}

/**
 * 获取严重级别CSS类
 * @param {string} severity - 严重级别
 * @returns {string} CSS类名
 */
function getSeverityClass(severity) {
  switch (severity) {
    case 'high': return 'severity-high';
    case 'medium': return 'severity-medium';
    case 'low': return 'severity-low';
    default: return 'severity-low';
  }
}

// ========== 时钟初始化 ==========

/**
 * 初始化实时时钟，每秒更新一次
 */
function initClock() {
  const clockEl = document.getElementById('clock');

  function updateClock() {
    clockEl.textContent = formatTime(new Date());
  }

  updateClock();
  setInterval(updateClock, 1000);
}

// ========== RUL趋势图初始化 ==========

/**
 * 初始化RUL趋势预测图表
 */
function initRULChart() {
  const chartDom = document.getElementById('rul-chart');
  rulChart = echarts.init(chartDom, 'dark');

  // 图表配置选项
  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(22, 33, 62, 0.95)',
      borderColor: '#1e3a5f',
      textStyle: { color: '#e0e6ed' },
      formatter: function(params) {
        let result = `<strong>${params[0].axisValue}</strong><br/>`;
        params.forEach(p => {
          const color = p.seriesName === '预测RUL' ? '#3b82f6' : '#10b981';
          result += `<span style="color:${color}">●</span> ${p.seriesName}: <strong>${p.value.toFixed(1)}</strong> 周期<br/>`;
        });
        return result;
      }
    },
    legend: {
      data: ['预测RUL', '实际RUL'],
      textStyle: { color: '#8899aa' },
      top: 5,
      right: 20
    },
    grid: {
      left: 50,
      right: 20,
      top: 40,
      bottom: 30,
      containLabel: false
    },
    xAxis: {
      type: 'category',
      data: [],
      axisLine: { lineStyle: { color: '#1e3a5f' } },
      axisLabel: { color: '#8899aa', fontSize: 11 },
      splitLine: { show: false }
    },
    yAxis: {
      type: 'value',
      name: 'RUL (周期)',
      nameTextStyle: { color: '#8899aa', fontSize: 11 },
      min: 0,
      max: 100,
      axisLine: { lineStyle: { color: '#1e3a5f' } },
      axisLabel: { color: '#8899aa', fontSize: 11 },
      splitLine: { lineStyle: { color: 'rgba(30, 58, 95, 0.5)', type: 'dashed' } }
    },
    series: [
      {
        name: '预测RUL',
        type: 'line',
        smooth: true,
        symbol: 'circle',
        symbolSize: 6,
        lineStyle: { color: '#3b82f6', width: 2 },
        itemStyle: { color: '#3b82f6' },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
            { offset: 1, color: 'rgba(59, 130, 246, 0.02)' }
          ])
        },
        data: []
      },
      {
        name: '实际RUL',
        type: 'line',
        smooth: true,
        symbol: 'diamond',
        symbolSize: 8,
        lineStyle: { color: '#10b981', width: 2 },
        itemStyle: { color: '#10b981' },
        data: []
      }
    ]
  };

  rulChart.setOption(option);

  // 响应窗口大小变化
  window.addEventListener('resize', () => {
    rulChart.resize();
  });
}

/**
 * 更新RUL图表数据
 * @param {Object} data - 包含timeLabels, predictedRUL, trueRUL的数据对象
 */
function updateRULChart(data) {
  if (!rulChart) return;

  const option = {
    xAxis: { data: data.timeLabels || [] },
    series: [
      { name: '预测RUL', data: data.predictedRUL || [] },
      { name: '实际RUL', data: data.trueRUL || [] }
    ]
  };

  rulChart.setOption(option);
}

// ========== 传感器仪表盘初始化 ==========

/**
 * 初始化4个传感器仪表盘
 */
function initGauges() {
  // 温度仪表
  gaugeTemp = echarts.init(document.getElementById('gauge-temp'), 'dark');
  // 振动仪表
  gaugeVib = echarts.init(document.getElementById('gauge-vib'), 'dark');
  // 电流仪表
  gaugeCurr = echarts.init(document.getElementById('gauge-curr'), 'dark');
  // 转速仪表
  gaugeRpm = echarts.init(document.getElementById('gauge-rpm'), 'dark');

  // 绑定响应式
  window.addEventListener('resize', () => {
    gaugeTemp.resize();
    gaugeVib.resize();
    gaugeCurr.resize();
    gaugeRpm.resize();
  });

  // 初始化空仪表
  initEmptyGauge(gaugeTemp, '温度 (°C)', 0, 100);
  initEmptyGauge(gaugeVib, '振动 (mm/s)', 0, 20);
  initEmptyGauge(gaugeCurr, '电流 (A)', 0, 50);
  initEmptyGauge(gaugeRpm, '转速 (RPM)', 0, 3000);
}

/**
 * 初始化空仪表盘
 * @param {Object} chart - ECharts实例
 * @param {string} name - 仪表名称
 * @param {number} min - 最小值
 * @param {number} max - 最大值
 */
function initEmptyGauge(chart, name, min, max) {
  const option = {
    backgroundColor: 'transparent',
    series: [{
      type: 'gauge',
      name: name,
      startAngle: 200,
      endAngle: -20,
      min: min,
      max: max,
      radius: '85%',
      splitNumber: 5,
      axisLine: {
        lineStyle: {
          width: 12,
          color: [[0.6, '#10b981'], [0.8, '#f59e0b'], [1, '#ef4444']]
        }
      },
      pointer: {
        itemStyle: { color: '#e0e6ed' },
        width: 4,
        length: '60%'
      },
      axisTick: {
        distance: -15,
        length: 5,
        lineStyle: { color: '#8899aa', width: 1 }
      },
      splitLine: {
        distance: -18,
        length: 10,
        lineStyle: { color: '#8899aa', width: 2 }
      },
      axisLabel: {
        color: '#8899aa',
        distance: 20,
        fontSize: 10
      },
      detail: {
        valueAnimation: true,
        formatter: '{value}',
        color: '#e0e6ed',
        fontSize: 14,
        fontWeight: 'bold',
        offsetCenter: [0, '70%']
      },
      title: {
        offsetCenter: [0, '90%'],
        color: '#8899aa',
        fontSize: 11
      },
      data: [{ value: 0, name: name }]
    }]
  };

  chart.setOption(option);
}

/**
 * 更新传感器仪表盘数据
 * @param {Object} data - 包含temp, vib, curr, rpm的数据对象
 */
function updateGauges(data) {
  if (!gaugeTemp) return;

  // 温度仪表
  gaugeTemp.setOption({
    series: [{
      data: [{ value: data.temp || 0, name: '温度 (°C)' }]
    }]
  });

  // 振动仪表
  gaugeVib.setOption({
    series: [{
      data: [{ value: data.vib || 0, name: '振动 (mm/s)' }]
    }]
  });

  // 电流仪表
  gaugeCurr.setOption({
    series: [{
      data: [{ value: data.curr || 0, name: '电流 (A)' }]
    }]
  });

  // 转速仪表
  gaugeRpm.setOption({
    series: [{
      data: [{ value: data.rpm || 0, name: '转速 (RPM)' }]
    }]
  });
}

// ========== 设备卡片渲染 ==========

/**
 * 渲染设备状态卡片列表
 * @param {Array} devices - 设备列表数据
 */
function renderDeviceCards(devices) {
  const container = document.getElementById('device-cards');
  container.innerHTML = '';

  devices.forEach(device => {
    const statusClass = getStatusClass(device.status);
    const rulVal = device.rul != null ? device.rul : 1.0;
    const rulPercent = Math.round(rulVal * 100);
    const rulClass = rulPercent >= 60 ? 'rul-high' : rulPercent >= 30 ? 'rul-medium' : 'rul-low';

    const card = document.createElement('div');
    card.className = `device-card ${statusClass}${selectedDeviceId === device.device_id ? ' selected' : ''}`;
    card.dataset.deviceId = device.device_id;
    card.onclick = () => selectDevice(device.device_id);

    const statusLabels = { normal: '正常', warning: '告警', danger: '故障' };
    const statusText = statusLabels[device.status] || device.status || '未知';

    card.innerHTML = `
      <div class="device-card-header">
        <span class="device-name">${device.device_name || device.device_id}</span>
        <span class="device-status-badge">${statusText}</span>
      </div>
      <div class="device-info">
        <span>设备ID</span>
        <span>${device.device_id}</span>
      </div>
      <div class="device-info">
        <span>健康得分</span>
        <span>${rulPercent}%</span>
      </div>
      <div class="rul-bar-container">
        <div class="rul-bar-label">
          <span>RUL</span>
          <span>${rulPercent}%</span>
        </div>
        <div class="rul-bar">
          <div class="rul-bar-fill ${rulClass}" style="width: ${rulPercent}%"></div>
        </div>
      </div>
    `;

    container.appendChild(card);
  });
}

// ========== 设备选择与详情获取 ==========

/**
 * 选中设备，更新图表显示
 * @param {string} deviceId - 设备ID
 */
function selectDevice(deviceId) {
  // 更新选中状态
  selectedDeviceId = deviceId;

  // 更新卡片选中样式
  document.querySelectorAll('.device-card').forEach(card => {
    card.classList.toggle('selected', card.dataset.deviceId === deviceId);
  });

  // 获取设备详情
  fetchDeviceDetails(deviceId);
}

/**
 * 获取并显示设备详细信息
 * @param {string} deviceId - 设备ID
 */
async function fetchDeviceDetails(deviceId) {
  try {
    // 并行获取遥测数据和RUL数据
    const [telemetryRes, rulRes] = await Promise.all([
      fetch(`${API_BASE}/api/device/${deviceId}/telemetry`),
      fetch(`${API_BASE}/api/device/${deviceId}/rul`)
    ]);

    if (telemetryRes.ok) {
      const telemetryData = await telemetryRes.json();
      updateGauges(telemetryData);
    }

    if (rulRes.ok) {
      const rulData = await rulRes.json();
      updateRULChart(rulData);
    }
  } catch (error) {
    console.error(`获取设备 ${deviceId} 详情失败:`, error);
    showToast(`设备数据加载失败: ${error.message}`, 'error');
  }
}

// ========== 告警列表渲染 ==========

/**
 * 渲染告警事件列表
 * @param {Array} alerts - 告警列表数据
 */
function renderAlerts(alerts) {
  const tbody = document.getElementById('alerts-tbody');
  tbody.innerHTML = '';

  if (!alerts || alerts.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#8899aa;padding:20px;">暂无告警记录</td></tr>';
    return;
  }

  alerts.forEach(alert => {
    const tr = document.createElement('tr');

    tr.innerHTML = `
      <td>${alert.timestamp || '-'}</td>
      <td>${alert.device_id || '-'}</td>
      <td>${alert.fault_type || '-'}</td>
      <td><span class="severity-badge ${getSeverityClass(alert.severity)}">${alert.severity_text || alert.severity}</span></td>
      <td><button class="btn-diagnosis" onclick="showDiagnosis('${alert.device_id}')">AI诊断</button></td>
    `;

    tbody.appendChild(tr);
  });
}

// ========== AI诊断报告弹窗 ==========

let lastFocusedElement = null;

/**
 * 显示AI诊断报告弹窗
 * @param {string} deviceId - 设备ID
 */
async function showDiagnosis(deviceId) {
  const overlay = document.getElementById('modal-overlay');
  const modalBody = document.getElementById('modal-body');
  const modalTitle = document.getElementById('modal-title');

  // 保存当前焦点元素，弹窗关闭时恢复
  lastFocusedElement = document.activeElement;

  modalTitle.textContent = `AI诊断报告 - ${deviceId}`;
  modalBody.innerHTML = `<div style="text-align:center;padding:40px;color:#8899aa;"><span class="loading-spinner"></span>正在生成诊断报告...</div>`;
  overlay.classList.add('active');

  // 弹窗打开后，将焦点移到弹窗标题（可访问性）
  modalTitle.setAttribute('tabindex', '-1');
  modalTitle.focus();

  try {
    const response = await fetch(`${API_BASE}/api/device/${deviceId}/diagnosis`);
    if (!response.ok) throw new Error('获取诊断报告失败');
    const report = await response.json();

    // 后端返回: fault_analysis, steps[], parts_needed[], severity, source, device_id, timestamp
    const sev = report.severity || 'low';
    const sevBadgeClass = sev === 'high' ? 'severity-high' : sev === 'medium' ? 'severity-medium' : 'severity-low';
    const sevLabel = { high: '高', medium: '中', low: '低' }[sev] || sev;
    const sourceLabel = report.source === 'llm' ? 'AI大模型' : '模板匹配';
    const stepsHtml = Array.isArray(report.steps) && report.steps.length
      ? '<ol style="padding-left:20px;margin:8px 0;">' + report.steps.map(s => `<li style="margin:6px 0;color:var(--text-secondary)">${s}</li>`).join('') + '</ol>'
      : '<p style="color:var(--text-secondary)">暂无步骤信息</p>';
    const partsHtml = Array.isArray(report.parts_needed) && report.parts_needed.length
      ? '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;">' + report.parts_needed.map(p => `<span style="background:rgba(16,185,129,0.15);color:#10b981;padding:4px 10px;border-radius:4px;font-size:0.8rem;">${p}</span>`).join('') + '</div>'
      : '<p style="color:var(--text-secondary)">暂无备件信息</p>';

    modalBody.innerHTML = `
      <h4 style="color:var(--accent-cyan);font-size:0.95rem;margin:16px 0 8px;padding-bottom:4px;border-bottom:1px dashed var(--border-color)">基本信息</h4>
      <ul style="list-style:none;padding:0;margin:8px 0;color:var(--text-secondary)">
        <li style="margin:4px 0"><strong style="color:var(--text-primary);display:inline-block;width:80px">设备ID:</strong> ${report.device_id || deviceId}</li>
        <li style="margin:4px 0"><strong style="color:var(--text-primary);display:inline-block;width:80px">严重程度:</strong> <span class="${sevBadgeClass}" style="padding:2px 8px;border-radius:4px;font-size:0.8rem;font-weight:600">${sevLabel}</span></li>
        <li style="margin:4px 0"><strong style="color:var(--text-primary);display:inline-block;width:80px">数据来源:</strong> ${sourceLabel}</li>
        <li style="margin:4px 0"><strong style="color:var(--text-primary);display:inline-block;width:80px">诊断时间:</strong> ${report.timestamp ? new Date(report.timestamp).toLocaleString('zh-CN') : '-'}</li>
      </ul>

      <h4 style="color:var(--accent-cyan);font-size:0.95rem;margin:16px 0 8px;padding-bottom:4px;border-bottom:1px dashed var(--border-color)">故障根因分析</h4>
      <div style="background:rgba(239,68,68,0.1);border-left:3px solid var(--accent-red);padding:12px;border-radius:4px;margin:8px 0;font-size:0.9rem;line-height:1.7;color:var(--text-primary)">${report.fault_analysis || '暂无分析结果'}</div>

      <h4 style="color:var(--accent-cyan);font-size:0.95rem;margin:16px 0 8px;padding-bottom:4px;border-bottom:1px dashed var(--border-color)">建议操作步骤</h4>
      ${stepsHtml}

      <h4 style="color:var(--accent-cyan);font-size:0.95rem;margin:16px 0 8px;padding-bottom:4px;border-bottom:1px dashed var(--border-color)">所需备件清单</h4>
      ${partsHtml}
    `;
  } catch (error) {
    console.error('获取诊断报告失败:', error);
    showToast(`诊断报告加载失败: ${error.message}`, 'error');
    modalBody.innerHTML = `
      <div style="text-align:center;padding:40px;color:#ef4444">
        <p>获取诊断报告失败</p>
        <p style="font-size:0.85rem;color:#8899aa;margin-top:8px">${error.message}</p>
      </div>
    `;
  }
}

function closeModal() {
  const overlay = document.getElementById('modal-overlay');
  overlay.classList.remove('active');
  // 恢复焦点到之前保存的元素（可访问性）
  if (lastFocusedElement && lastFocusedElement.focus) {
    lastFocusedElement.focus();
  }
}

/**
 * 键盘陷阱：确保 Tab 键在弹窗内循环
 */
function handleTabKey(e) {
  const overlay = document.getElementById('modal-overlay');
  if (!overlay.classList.contains('active')) return;

  const focusableElements = overlay.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  if (focusableElements.length === 0) return;

  const firstEl = focusableElements[0];
  const lastEl = focusableElements[focusableElements.length - 1];

  if (e.shiftKey) {
    // Shift+Tab: 如果焦点在第一个元素，移到最后一个
    if (document.activeElement === firstEl) {
      e.preventDefault();
      lastEl.focus();
    }
  } else {
    // Tab: 如果焦点在最后一个元素，移到第一个
    if (document.activeElement === lastEl) {
      e.preventDefault();
      firstEl.focus();
    }
  }
}

/**
 * 处理 Escape 键关闭弹窗
 */
function handleEscapeKey(e) {
  if (e.key === 'Escape') {
    const overlay = document.getElementById('modal-overlay');
    if (overlay.classList.contains('active')) {
      closeModal();
    }
  }
}

// ========== 数据轮询 ==========

/**
 * 从服务器获取最新数据
 */
async function fetchData() {
  try {
    // 获取设备列表
    const devicesRes = await fetch(`${API_BASE}/api/devices`);
    if (devicesRes.ok) {
      const devices = await devicesRes.json();
      renderDeviceCards(devices);

      // 如果有设备且未选中任何设备，自动选中第一个
      if (devices.length > 0 && !selectedDeviceId) {
        selectDevice(devices[0].id);
      } else if (selectedDeviceId) {
        // 刷新当前选中设备的详情
        fetchDeviceDetails(selectedDeviceId);
      }
    }

    // 获取告警列表
    const alertsRes = await fetch(`${API_BASE}/api/alerts`);
    if (alertsRes.ok) {
      const alerts = await alertsRes.json();
      renderAlerts(alerts);
    }
  } catch (error) {
    console.error('数据获取失败:', error);
  }
}

// ========== 初始化 ==========

/**
 * 页面加载完成后初始化所有组件
 */
document.addEventListener('DOMContentLoaded', () => {
  // 初始化时钟
  initClock();

  // 初始化RUL趋势图
  initRULChart();

  // 初始化传感器仪表盘
  initGauges();

  // 绑定关闭弹窗事件
  document.getElementById('modal-close').addEventListener('click', closeModal);
  document.getElementById('modal-overlay').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeModal();
  });

  // 键盘事件：Escape 关闭弹窗，Tab 陷阱
  document.addEventListener('keydown', (e) => {
    handleEscapeKey(e);
    handleTabKey(e);
  });

  // 首次获取数据
  fetchData();

  // 启动定时轮询
  setInterval(fetchData, POLL_INTERVAL);
});