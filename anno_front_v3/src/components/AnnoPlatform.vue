<template>
    <div id='test' >
        <!-- div id='select' style="width:1600px;" align="center">
        </div> -->
        <div id='status' style='width:800px;height:52px;position:absolute;top:12px;left:132px;' align='center'></div>
        <div id='function_block' style='width:116px;position:absolute;top:64px;left:16px;' align='center'>
            <div class="button_box">
                <button class='button' id='add'>多边形标注(a)</button>
                <button class='button' id='bbox'>矩形标注(b)</button>
                <button class='button' id='line'>线段标注(l)</button>
                <button class='button' id='sam_pred'>SAM标注(m)</button>
                <button class='button' id='undo'>回退一步(u)</button>
                <button class='button' id='close'>完成标注(c)</button>
                <button class='button' id='quit'>退出标注(q)</button>
                <button class='button' id='save'>保存编辑(s)</button>
                <button class='button' id='download_anno'>下载标注</button>
                <button class='button' id='download_fg'>下载抠图</button>
            </div>
            <br>
            <div class="button_box">
                <button class='button' id='insert'>增加顶点(i)</button>
                <button class='button' id='pop'>删除顶点(p)</button>
                <button class='button' id='delete'>删除标注(d)</button>
                <label class='small_text'>显示标注</label>
                <input class='mui-switch' type='checkbox' id='show_anno' checked>
            </div>
            <br>
            <div class="button_box">
                <button class='button' id='region'>放大(r)</button>
                <button class='button' id='reset'>重置(q)</button>
            </div>
        </div>
        <div id='canvas' style='width:800px;height:800px;position:absolute;top:64px;left:132px;' align='center'>
            <canvas id='bg' width=768 height=768></canvas>
        </div>
        <!-- <div class='button_box' style='width:320px;position:absolute;top:64px;left:932px;' align='center'></div> -->
        <div class='button_box' style='width:320px;position:absolute;top:64px;left:932px;' align='center'>
            <label class='text'>选择模型</label>
            <select id='sam_selecter' class='up_selecter'>
                <option value='vit_h'>SAM-H</option>
                <option value='vit_b' selected>SAM-B</option>
                <!-- <option value='vit-t'>SAM-T</option> -->
                <option value='med-vit_b'>MedSAM-B</option>
                <option value='UNI'>UNI</option>
            </select>
            <br>
            <label class='text'>选择序列</label>
            <select id='collection_selecter' class='up_selecter'>
                <option value="medical.png">medical</option>
            </select>
            <br>
            <label class='text'>选择图像</label>
            <select id='image_selecter' class='up_selecter'>
                <option value="medical.png">medical</option>
            </select>
            <br>
            <label class='text'>滑动选择图像</label>
            <input class='select_range' type='range' id="select_image" min="0" max="100" />
            <br>
            <div class="container">
                <label class='text'>点击选择</label>
                &nbsp;&nbsp;&nbsp;&nbsp;
                <button class='small_button' id='prev'>上一张图像</button>
                <button class='small_button' id='next'>下一张图像</button>
            </div>
        </div>
        <div class='button_box' style='width:320px;position:absolute;top:320px;left:932px;' align='center'>
            <button class='small_button' id='dilate_anno'>扩充实例</button>
            <button class='small_button' id='erode_anno'>删除假阳</button>
            <button class='button' id='clear_anno'>清空标注</button>
            <label class='small_text'>扩充数量</label>
            <input class='small_text' type='text' id='dilate_max_objs' value="100" />
            &nbsp;
            <label class='small_text'>扩充阈值</label>
            <input class='select_range' type='range' id='dilate_thres' min='0' max='10' default='5' />
        </div>
        <div class='button_box' style='width:320px;position:absolute;top:420px;left:932px;' align='center'>
            <input class='small_button' type='button' value='上传图片' onclick="document.getElementById('upload').click()">
            <input type="file" id="upload" accept="image/png, image/jpeg" name="upload" style="display:none" readonly/>
            <input class='small_button' type='button' value='上传标注' onclick="document.getElementById('upload_json').click()">
            <input type="file" id="upload_json" accept="application/json" name="upload_json" style="display:none" readonly/>
            <button class='small_button' id='calc_volume'>各类别面积</button>
            <button class='small_button' id='rename'>重命名</button>
            <button class='small_button' id='remove'>删除图片</button>
            <button class='small_button' id='instruct'>使用说明</button>
            <label class='text'>SAM预测结果平滑程度</label>
            <input class='select_range' type='range' id='comp_force' min='0' max='5' default='2' />
        </div>
        <div class="button_box" style='width:320px;position:absolute;top:580px;left:932px;' align='center'>
            <label class='text'>标签颜色表</label>
            <div class='color-table'>
                <div class="color" id='color_1' style="background-color: #D79B00;">1</div>
                <div class="color" id='color_2' style="background-color: #6C8EBF;">2</div>
                <div class="color" id='color_3' style="background-color: #82B366;">3</div>
                <div class="color" id='color_4' style="background-color: #B85450;">4</div>
                <div class="color" id='color_5' style="background-color: #9673A6;">5</div>
                <div class="color" id='color_6' style="background-color: #FFFF88;">6</div>
            </div>
        </div>
        <div id='debug_block' style='width:384px;position:absolute;top:512px;left:932px;' align='center'></div>
        <!-- <img :src='image_url' /> -->
    </div>
</template>

<script lang='ts'>

// Tools and functions
import axios from 'axios'
// declare const Buffer: any


function min(a: number, b: number) {
    return a < b ? a : b
}

function max(a: number, b: number) {
    return a > b ? a : b
}

let backend_address: string = 'http://162.105.89.250:18002'

let collection_name: string = ''   // collection name
let token: string = ''  // token for authentification

// Block right
function blockright(oEvent: any) {
    if (window.event) {
        oEvent = window.event
        oEvent.returnValue = false
    } else {
        oEvent.preventDefault()
    }
}
window.onload = () => {document.oncontextmenu = blockright}

function hexToRGBA(hex: string, alpha: number): string | null {
    const regex = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i
    const result = regex.exec(hex)
    if (!result) {
        return null
    }
    const [, r, g, b] = result
    const red = parseInt(r, 16)
    const green = parseInt(g, 16)
    const blue = parseInt(b, 16)
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`
}

function throttle(fn: any, delay: number) {
    let prev = 0
    return (...args: any[]) => {
        let now = new Date().getTime()
        if (now - prev >= delay) {
            prev = now
            fn(...args)
        }
    }
}

let this_bg: any
let this_context: any
let this_annotations = [] as any
let this_annotation = {} as any
let this_sam_annotations = [] as any
let this_anno_idx_map: any
let this_is_bbox_map: any
let this_is_line_map: any
let this_path_idx_map: any
let this_center_map: any
// let needReDrawImageAnno = false
// let animationFrameHandle: number | null = null
// The main class
export default {

    data() {
        return {
            collection_selecter: undefined as any,    // collection_selecter component
            image_selecter: undefined as any,     // image_selecter component
            sam_selecter: undefined as any,       // sam_selecter component
            status: undefined as any,             // status component
            // bg: undefined as any,                 // bg component
            // context: undefined as any,            // context component
            image_name: '', // image name
            image: new Image(),     // 'image' instance
            image_url: '233',       // filename
            // annotations: [] as any,        // a list of dict
            // annotation: {} as any,         // a dict
            bbox: {} as any,               // bounding box
            line: [] as any,               // line
            //
            prompt: {'points': [], 'labels': [], 'bbox': {}} as any,
            // sam_annotations: [], // SAM的预测结果
            sam_set_image_timer: undefined as any,  // SAM提前加载图片的倒计时
            //
            drag_index: -1, // index of the selected annotation
            point_idx: 0,   // index of the selected point
            region_info: {} as any,   // information of selected region
            image_list: [] as any,    // image list
            collection_list: [] as any,   // collection list
            //
            detaildiv: undefined as any,          // Detail Div
            debugdiv: undefined as any,           // Debug Div

            // two-dimentional arrays
            // anno_idx_map: undefined as any,
            // is_bbox_map: undefined as any,
            // path_idx_map: undefined as any,
            // center_map: undefined as any,

            // function modes
            drag_mode: false,          // drag points
            drag_bbox: false,          // are you dragging a bbox now?
            draw_bbox_mode: false,     // draw a bbox
            drag_line: false,           // are you dragging a line now?
            draw_line_mode: false,     // draw a line
            draw_polygon_mode: false,  // draw a polygon
            delete_mode: false,        // delete an annotation
            pop_mode: false,           // pop a point from a polygon
            insert_mode: false,        // insert a point into a polygon
            region_mode: false,        // select a region
            selecting: false,          // selecting a region
            prompt_mode: false,  // SAM的point prompt
            //
            base_point_size: 4,     // base size of a point
            point_size: 0,          // size of a point
            receptive_ratio: 2,     // ratio of receptive field
            line_width: 1,          // size of line
            // style_list: ['#D79B00', '#6C8EBF', '#82B366', '#B85450', '#9673A6', '#FFFF88']
            style_list: ['#FF0000', '#00FF00', '#0000FF', '#800080', '#FFA500', '#FFFF00'],
            // style_list: []
            // default_style: '#0197F6'
            // stroke_style: '#ff0033'    // color of lines
            upload_button: undefined as any,          // upload_button
            upload_json_button: undefined as any,     // upload_json_button
            //
            dilate_max_objs: undefined as any,
            dilate_thres: undefined as any,
            comp_force_range: undefined as any,
            show_anno: undefined as any,
            // adjust_force_range: undefined as any,
            select_image_range: undefined as any,

            // descriptions for all of the modes
            descriptions: {
                'drag_mode': '鼠标左键点击顶点可进行拖拽，按s键保存标注编辑结果',
                'draw_polygon_mode': '鼠标左键创造标注，鼠标右键或c键完成标注，u键回退一步，q键退出',
                'draw_bbox_mode': '鼠标左键选择第一个点，再点一次选择第二个点，q键退出',
                'draw_line_mode': '鼠标左键选择第一个点，再点一次选择第二个点，q键退出',
                'delete_mode': '通过点击任意顶点可以删除指定标注，q键退出当前模式',
                'pop_mode': '通过点击任意多边形顶点删除指定顶点，q键退出当前模式',
                'insert_mode': '通过点击任意多边形顶点在附近增加顶点，q键退出当前模式',
                'region_mode': '鼠标左键选择矩形框的两个边界点以指定区域，q键退出当前模式',
                'prompt_mode': '鼠标左键选择前景点，右键选择背景点，c键完成标注，u键回退，q键退出',
            },
        }
    },

    methods: {
        SAMPredict() {
            if (!('xmin' in this.prompt['bbox']) && (this.prompt['points'].length == 0)) {
                this_sam_annotations = []
                this.drawImageAnno()
                return
            }
            this.status.innerHTML = '预测中，请稍后'
            axios
                .post(
                    backend_address + '/get_sam_pred',
                    {
                        'token': token,
                        'sam_type': this.sam_selecter.options[this.sam_selecter.selectedIndex].value,
                        'collection_name': collection_name,
                        'image_name': this.image_name,
                        'compress_degree': parseInt(this.comp_force_range.value, 10),
                        'prompt_points': this.prompt['points'],
                        'prompt_labels': this.prompt['labels'],
                        'prompt_bbox': this.prompt['bbox'],
                    },
                )
                .then(response => {
                    this.status.innerHTML = this.descriptions['prompt_mode']
                    this_sam_annotations = response.data
                    this.drawImageAnno()
                })
                .catch(error => {
                    this.status.innerHTML = this.descriptions['prompt_mode']
                    console.log(error)
                })
                .finally(() => {})
        },

        dilateAnno() {
            this.status.innerHTML = '预测中，请稍后'
            axios
                .post(
                    backend_address + '/get_more_sam_pred',
                    {
                        'token': token,
                        'sam_type': this.sam_selecter.options[this.sam_selecter.selectedIndex].value,
                        'collection_name': collection_name,
                        'image_name': this.image_name,
                        'compress_degree': parseInt(this.comp_force_range.value, 10),
                        'max_objs': parseInt(this.dilate_max_objs.value, 10),
                        'sim_thres': parseInt(this.dilate_thres.value, 10) / 10,
                    },
                )
                .then(response => {
                    this.status.innerHTML = this.descriptions['drag_mode']
                    this_annotation = {}
                    this_annotations = response.data
                    this.drawImageAnno()
                })
                .catch(error => {
                    this.status.innerHTML = this.descriptions['drag_mode']
                    console.log(error)
                })
                .finally(() => {})
        },

        erodeAnno() {
            this.status.innerHTML = '预测中，请稍后'
            axios
                .post(
                    backend_address + '/get_less_sam_pred',
                    {
                        'token': token,
                        'sam_type': this.sam_selecter.options[this.sam_selecter.selectedIndex].value,
                        'collection_name': collection_name,
                        'image_name': this.image_name,
                        'compress_degree': parseInt(this.comp_force_range.value, 10),
                        'anno': this_annotations,
                    },
                )
                .then(response => {
                    this.status.innerHTML = this.descriptions['drag_mode']
                    this_annotation = {}
                    this_annotations = response.data
                    this.drawImageAnno()
                })
                .catch(error => {
                    this.status.innerHTML = this.descriptions['drag_mode']
                    console.log(error)
                })
                .finally(() => {})
        },

        resetRegion() {
            this.region_info = {
                'xmin': 0, 'ymin': 0,
                'width': this.image.width,
                'height': this.image.height,
                'ratio': min(
                    this_bg.height / (this.image.height),
                    this_bg.width / (this.image.width),
                ),
            }
            this.point_size = this.base_point_size / this.region_info['ratio']
            console.log('Reset region')
            console.log('bg.height = ', this_bg.height, '; bg.width = ', this_bg.width)
            console.log('image.height = ', this.image.height, '; image.width = ', this.image.width)
            console.log('region_info = ', this.region_info['ratio'])
        },

        checkNode(x: number, y: number) {
            let xmin: number = 1
            let ymin: number = 1
            let xmax: number = (this.region_info['width'] - 1) * this.region_info['ratio']
            let ymax: number = (this.region_info['height'] - 1) * this.region_info['ratio']
            return (
                x >= xmin && x <= xmax
                && y >= ymin && y <= ymax
            )
        },

        // Overwritten functions
        // x => (x - region_info['xmin']) * region_info['ratio']
        // y => (y - region_info['ymin']) * region_info['ratio']
        // width => width * region_info['ratio']
        // height => height * region_info['ratio']
        strokeRect(xmin: number, ymin: number, width: number, height: number) {
            if (!this.show_anno.checked) {
                return
            }
            xmin = (xmin - this.region_info['xmin']) * this.region_info['ratio']
            ymin = (ymin - this.region_info['ymin']) * this.region_info['ratio']
            width = width * this.region_info['ratio']
            height = height * this.region_info['ratio']
            if (this.checkNode(xmin, ymin)) {
                this_context.strokeRect(xmin, ymin, width, height)
            }
        },

        fillRect(xmin: number, ymin: number, width: number, height: number) {
            if (!this.show_anno.checked) {
                return
            }
            xmin = (xmin - this.region_info['xmin']) * this.region_info['ratio']
            ymin = (ymin - this.region_info['ymin']) * this.region_info['ratio']
            width = width * this.region_info['ratio']
            height = height * this.region_info['ratio']
            if (this.checkNode(xmin, ymin)) {
                this_context.fillRect(xmin, ymin, width, height)
            }
        },

        fillStar(x: number, y: number, star_label: number) {
            if (!this.show_anno.checked) {
                return
            }
            x = (x - this.region_info['xmin']) * this.region_info['ratio']
            y = (y - this.region_info['ymin']) * this.region_info['ratio']
            this_context.font = '16px Arial'
            this_context.fillStyle = '#000000'
            if (star_label == 1) {
                this_context.fillText('\uD83D\uDD25', x, y)
            } else {
                this_context.fillText('\u2744️', x, y)
            }
        },

        checkPath(path: any) {
            let flag: boolean = true
            let x: number
            let y: number
            let xmin: number = 1
            let ymin: number = 1
            let xmax: number = (this.region_info['width'] - 1) * this.region_info['ratio']
            let ymax: number = (this.region_info['height'] - 1) * this.region_info['ratio']
            let valid_sub_path = []
            for (let j = 0; j < path.length; j++) {
                x = (path[j]['x'] - this.region_info['xmin']) * this.region_info['ratio']
                y = (path[j]['y'] - this.region_info['ymin']) * this.region_info['ratio']
                if (x < xmin) {x = xmin; flag = false}
                if (x > xmax) {x = xmax; flag = false}
                if (y < ymin) {y = ymin; flag = false}
                if (y > ymax) {y = ymax; flag = false}
                valid_sub_path.push(
                    {
                        'x': x / this.region_info['ratio'] + this.region_info['xmin'],
                        'y': y / this.region_info['ratio'] + this.region_info['ymin'],
                    },
                )
            }
            return {'flag': flag, 'valid_sub_path': valid_sub_path}
        },

        moveTo(x: number, y: number) {
            if (!this.show_anno.checked) {
                return
            }
            x = (x - this.region_info['xmin']) * this.region_info['ratio']
            y = (y - this.region_info['ymin']) * this.region_info['ratio']
            this_context.beginPath()
            this_context.moveTo(x, y)
        },

        lineTo(x: number, y: number) {
            if (!this.show_anno.checked) {
                return
            }
            x = (x - this.region_info['xmin']) * this.region_info['ratio']
            y = (y - this.region_info['ymin']) * this.region_info['ratio']
            this_context.lineTo(x, y)
        },

        closePath() {
            if (!this.show_anno.checked) {
                return
            }
            this_context.closePath()
            let prev_fill_style = this_context.fillStyle
            this_context.fillStyle = hexToRGBA(prev_fill_style, 0.3)
            this_context.fill()
            this_context.fillStyle = prev_fill_style
        },

        stroke() {
            if (!this.show_anno.checked) {
                return
            }
            this_context.stroke()
        },

        maskAllModes() {
            // mode flags
            this.draw_bbox_mode = false
            this.draw_line_mode = false
            this.draw_polygon_mode = false
            this.drag_mode = false
            this.delete_mode = false
            this.pop_mode = false
            this.insert_mode = false
            this.region_mode = false
            this.prompt_mode = false
            // this.bbox_pred_mode = false
            // other flags
            this.drag_bbox = false
            this.drag_line = false
            this.selecting = false
            this.status.innerHTML = this.descriptions['drag_mode']
        },

        checkAllModes() {
            // return !(
            //     this.draw_bbox_mode || this.draw_polygon_mode || this.drag_mode
            //     || this.delete_mode || this.pop_mode
            //     || this.insert_mode || this.region_mode
            // )
            return !(this.draw_polygon_mode || this.draw_bbox_mode || this.draw_line_mode || this.region_mode || this.prompt_mode)
        },

        setStyle(label: string) {
            let int_label = parseInt(label, 10)
            if (isNaN(int_label) || int_label < 1 || int_label > this.style_list.length) {
                int_label = 1
            }
            // console.log('Set style', this.style_list[int_label - 1])
            this_context.fillStyle = this.style_list[int_label - 1]
            this_context.strokeStyle = this.style_list[int_label - 1]
        },

        resetStyle() {
            this_context.fillStyle = this.style_list[0]
            this_context.strokeStyle = this.style_list[0]
        },

        clearCanvas() {
            // eslint-disable-next-line
            this_bg.height = this_bg.height // clear the canvas
            // this_context.fillStyle = this.fill_style
            // this_context.strokeStyle = this.stroke_style
            this_context.lineWidth = this.line_width
        },

        // drawImageAnno() {
        //     if (animationFrameHandle === null) {
        //         this._drawImageAnno()
        //     } else {
        //         needReDrawImageAnno = true
        //     }
        // },

        drawImageAnno() {
            this.clearCanvas()
            this_context.drawImage(
                this.image,
                this.region_info['xmin'], this.region_info['ymin'],
                this.region_info['width'], this.region_info['height'],
                0, 0, this.region_info['width'] * this.region_info['ratio'],
                this.region_info['height'] * this.region_info['ratio'],
            )
            this.resetStyle()
            for (let i = 0; i < this_annotations.length; i++) {
                // 画点
                let annotation: any = this_annotations[i]
                this.setStyle(annotation['label'])
                if ('path' in annotation) {
                    // 多边形
                    // 1.画点
                    for (let j = 0; j < annotation['path'].length; j++) {
                        this.fillRect(
                            annotation['path'][j]['x'] - this.point_size / 2,
                            annotation['path'][j]['y'] - this.point_size / 2,
                            this.point_size,
                            this.point_size,
                        )
                        this.fillMap(
                            annotation['path'][j]['x'],
                            annotation['path'][j]['y'],
                            this.point_size * this.receptive_ratio,
                            i, false, false, j,
                        )
                    }
                    // 2.画线
                    let valid_sub_path = this.checkPath(annotation['path'])['valid_sub_path']
                    if (valid_sub_path.length >= 3) {
                        this.setStyle(annotation['label'])
                        this.moveTo(
                            valid_sub_path[0]['x'],
                            valid_sub_path[0]['y'],
                        )
                        for (let j = 1; j < valid_sub_path.length; j++) {
                            this.lineTo(
                                valid_sub_path[j]['x'],
                                valid_sub_path[j]['y'],
                            )
                        }
                        this.closePath()
                        this.stroke()
                    }
                } else if ('bbox' in annotation) {
                    // 框
                    // 1.画点
                    let x_keys = ['xmin', 'xmax']
                    let y_keys = ['ymin', 'ymax']
                    for (let x_key_idx of [0, 1]) {
                        for (let y_key_idx of [0, 1]) {
                            let x_key: string = x_keys[x_key_idx]
                            let y_key: string = y_keys[y_key_idx]
                            this.fillRect(
                                annotation['bbox'][x_key] - this.point_size / 2,
                                annotation['bbox'][y_key] - this.point_size / 2,
                                this.point_size,
                                this.point_size,
                            )
                            this.fillMap(
                                annotation['bbox'][x_key],
                                annotation['bbox'][y_key],
                                this.point_size * this.receptive_ratio,
                                i, true, false, x_key_idx + y_key_idx * 2,
                            )
                        }
                    }
                    // 2.画线
                    let prev_fill_style = this_context.fillStyle
                    this_context.fillStyle = hexToRGBA(prev_fill_style, 0.3)
                    this.fillRect(
                        annotation['bbox']['xmin'],
                        annotation['bbox']['ymin'],
                        annotation['bbox']['xmax'] - annotation['bbox']['xmin'],
                        annotation['bbox']['ymax'] - annotation['bbox']['ymin'],
                    )
                    this_context.fillStyle = prev_fill_style
                } else if ('line' in annotation) {
                    // 线段
                    // 1.画点
                    this.fillRect(
                        annotation['line']['x0'] - this.point_size / 2,
                        annotation['line']['y0'] - this.point_size / 2,
                        this.point_size,
                        this.point_size,
                    )
                    this.fillMap(
                        annotation['line']['x0'],
                        annotation['line']['y0'],
                        this.point_size * this.receptive_ratio,
                        i, false, true, 0,
                    )
                    this.fillRect(
                        annotation['line']['x1'] - this.point_size / 2,
                        annotation['line']['y1'] - this.point_size / 2,
                        this.point_size,
                        this.point_size,
                    )
                    this.fillMap(
                        annotation['line']['x1'],
                        annotation['line']['y1'],
                        this.point_size * this.receptive_ratio,
                        i, false, true, 1,
                    )
                    // 2.画线
                    let prev_fill_style = this_context.fillStyle
                    let prev_line_width = this_context.lineWidth
                    this_context.fillStyle = hexToRGBA(prev_fill_style, 0.3)
                    this_context.lineWidth = prev_line_width + 1
                    this.moveTo(annotation['line']['x0'], annotation['line']['y0'])
                    this.lineTo(annotation['line']['x1'], annotation['line']['y1'])
                    this.stroke()
                    this_context.fillStyle = prev_fill_style
                    this_context.lineWidth = prev_line_width
                }
            }
            this.resetStyle()
            if ('xmin' in this.prompt['bbox']) {
                // 画SAM的prompt bbox
                this.strokeRect(
                    this.prompt['bbox']['xmin'],
                    this.prompt['bbox']['ymin'],
                    this.prompt['bbox']['xmax'] - this.prompt['bbox']['xmin'],
                    this.prompt['bbox']['ymax'] - this.prompt['bbox']['ymin'],
                )
            }
            for (let i = 0; i < this.prompt['points'].length; i++) {
                // 画SAM的prompt points
                this.fillStar(this.prompt['points'][i]['x'], this.prompt['points'][i]['y'], this.prompt['labels'][i])
            }
            for (let i = 0; i < this_sam_annotations.length; i++) {
                let annotation = this_sam_annotations[i]
                console.log('Annotation', i)
                let valid_sub_path = this.checkPath(annotation['path'])['valid_sub_path']
                if (valid_sub_path.length >= 3) {
                    this.setStyle('0')
                    this.moveTo(
                        valid_sub_path[0]['x'],
                        valid_sub_path[0]['y'],
                    )
                    for (let j = 1; j < valid_sub_path.length; j++) {
                        this.lineTo(
                            valid_sub_path[j]['x'],
                            valid_sub_path[j]['y'],
                        )
                    }
                    this.closePath()
                    this.stroke()
                }
            }
            // animationFrameHandle = requestAnimationFrame(() => {
            //     animationFrameHandle = null
            //     if (needReDrawImageAnno) {
            //         needReDrawImageAnno = false
            //         this._drawImageAnno()
            //     }
            // })
        },

        drawCurAnno() {
            if ('path' in this_annotation) {
                for (let j = 0; j < this_annotation['path'].length; j++) {
                    this.fillRect(
                        this_annotation['path'][j]['x'] - this.point_size / 2,
                        this_annotation['path'][j]['y'] - this.point_size / 2,
                        this.point_size,
                        this.point_size,
                    )
                }
            }
            if (
                'path' in this_annotation
                && this_annotation['path'].length > 0
                && this.checkPath(this_annotation['path'])['flag']
            ) {
                this.moveTo(
                    this_annotation['path'][0]['x'],
                    this_annotation['path'][0]['y'],
                )
                for (let j = 1; j < this_annotation['path'].length; j++) {
                    this.lineTo(
                        this_annotation['path'][j]['x'],
                        this_annotation['path'][j]['y'],
                    )
                }
            }
        },

        readJson(image_name: string) {
            return axios
                .post(
                    backend_address + '/get_anno',
                    {
                        'token': token,
                        'collection_name': collection_name,
                        'image_name': image_name,
                    },
                )
                .then(response => {
                    console.log('readJson', image_name)
                    this_annotations = response.data
                    this_annotation = {}
                    this.initialize()
                    this.resetRegion()
                    this.drawImageAnno()
                })
                .catch(error => {
                    console.log(error)
                    this_annotations = []
                    this_annotation = {}
                    this.initialize()
                    this.drawImageAnno()
                })
                .finally(() => {})
        },

        readImage() {
            console.log('readImage', this.image_name)
            return axios
                .post(
                    backend_address + '/get_image',
                    {
                        'token': token,
                        'collection_name': collection_name,
                        'image_name': this.image_name,
                    },
                )
                .then(response => {
                    // let encryptedBytes = Buffer.from(response.data)
                    let encryptedBytes = response.data
                    this.image_url = 'data:image/jpeg;base64,' + encryptedBytes.toString('base64')
                    this.image.src = this.image_url
                    const promise = this.readJson(this.image_name)
                    this.status.innerHTML = this.descriptions['drag_mode']
                    // 这个操作会降低切换图片的速度
                    if (this.image_name.length > 1) {
                        clearTimeout(this.sam_set_image_timer)
                        this.sam_set_image_timer = setTimeout(this.samSetImage, 1500)
                    }
                    // this.samSetImage()
                    return promise
                })
                .catch(error => {
                    console.log(error)
                })
                .finally(() => {})
        },

        samSetImage() {
            axios.post(
                backend_address + '/sam_set_image',
                {
                    'token': token,
                    'sam_type': this.sam_selecter.options[this.sam_selecter.selectedIndex].value,
                    'collection_name': collection_name,
                    'image_name': this.image_name,
                },
            )
            .then(response => {})
            .catch(error => {})
            .finally(() => {})
        },

        // update '<select>'
        updateCollectionSelecter() {
            let match_index = 0
            // update collection selecter
            this.collection_selecter.length = 0
            for (let item_name of this.collection_list) {
                let option = document.createElement("option")
                option.value = item_name
                option.text = item_name
                this.collection_selecter.add(option, null)
            }
            for (let i = 0; i < this.collection_selecter.length; i++) {
                if (this.collection_selecter.options[i].value == collection_name) {
                    match_index = i
                    break
                }
            }
            this.collection_selecter.selectedIndex = match_index
            collection_name = this.collection_selecter.options[this.collection_selecter.selectedIndex].value
        },

        updateImageSelecter() {
            // update image selecter
            let match_index = 0
            this.image_selecter.length = 0
            for (let image_info of this.image_list) {
                let option = document.createElement("option")
                option.value = image_info['image_name']
                option.text = image_info['image_name']
                this.image_selecter.add(option, null)
            }
            for (let i = 0; i < this.image_selecter.length; i++) {
                if (this.image_selecter.options[i].value == this.image_name) {
                    match_index = i
                    break
                }
            }
            this.image_selecter.selectedIndex = match_index
            this.image_name = this.image_selecter.options[this.image_selecter.selectedIndex].value
            // Read image
            return this.readImage()
        },

        nextImage() {
            let cur_idx = parseInt(this.select_image_range.value, 10)
            if (cur_idx < this.select_image_range.max) {
                cur_idx = cur_idx + 1
            }
            this.select_image_range.value = cur_idx
            this.image_selecter.selectedIndex = cur_idx
            this.image_name = this.image_selecter.options[cur_idx].value
            this.readImage()
        },

        prevImage() {
            let cur_idx = parseInt(this.select_image_range.value, 10)
            if (cur_idx > this.select_image_range.min) {
                cur_idx = cur_idx - 1
            }
            this.select_image_range.value = cur_idx
            this.image_selecter.selectedIndex = cur_idx
            this.image_name = this.image_selecter.options[cur_idx].value
            this.readImage()
        },

        calcVolume() {
            axios
                .post(
                    backend_address + '/calc_volume',
                    {
                        'token': token,
                        'collection_name': collection_name,
                        'image_name': this.image_name,
                    },
                )
                .then(response => {
                    alert(response.data)
                })
                .catch(error => {
                    console.log(error)
                })
                .finally(() => {})
        },

        onSelectImageChange(e: any) {
            this.image_selecter.selectedIndex = parseInt(this.select_image_range.value, 10)
            this.image_name = this.image_selecter.options[this.image_selecter.selectedIndex].value
            this.readImage()
        },

        onSelectImageMove(e: any) {
            this.detaildiv.innerHTML = this.select_image_range.value
            this.detaildiv.style.width = 12 + 12 * this.detaildiv.innerHTML.length + 'px'
            // console.log(this_annotations[this_anno_idx_map[y][x]])
            this.detaildiv.style.display = ''
            this.detaildiv.style.left = e.clientX + 16 + 'px'
            this.detaildiv.style.top = e.clientY + 16 + "px"
        },

        onSelectImageOut(e: any) {
            this.detaildiv.style.display = 'none'
        },

        onCollectionChange(e: any) {
            collection_name = this.collection_selecter.options[this.collection_selecter.selectedIndex].value
            this.image_name = ''
            this.readImageList()
        },

        onImageChange(e: any) {
            this.image_name = this.image_selecter.options[this.image_selecter.selectedIndex].value
            this.select_image_range.value = this.image_selecter.selectedIndex
            this.readImage()
        },

        readCollectionList() {
            console.log('Get collection list')
            return axios
                .post(backend_address + '/get_collection_list', {'token': token})
                .then(response => {
                    this.collection_list = response.data
                    this.updateCollectionSelecter()
                    return this.readImageList()
                })
                .catch(error => {
                    console.log(error)
                })
                .finally(() => {})
        },

        readImageList() {
            console.log('Get image list in collection:', collection_name)
            return axios
                .post(backend_address + '/get_image_list', {'token': token, 'collection_name': collection_name})
                .then(response => {
                    this.image_list = response.data
                    this.select_image_range.value = 0
                    this.select_image_range.min = 0
                    this.select_image_range.max = this.image_list.length - 1
                    return this.updateImageSelecter()
                })
                .catch(error => {
                    console.log(error)
                })
                .finally(() => {})
        },

        mouseMove(e: any) {
            this.drawImageAnno()
            let offset_x = e.offsetX / this.region_info['ratio'] + this.region_info['xmin']
            let offset_y = e.offsetY / this.region_info['ratio'] + this.region_info['ymin']
            let x = parseInt(offset_x + '', 10)
            let y = parseInt(offset_y + '', 10)
            if (x < 0 || y < 0 || x >= this.image.width || y >= this.image.height)
                return
            //
            if (this_anno_idx_map[y][x] != -1) {
                this.setStyle(this_annotations[this_anno_idx_map[y][x]]['label'])
                this.fillRect(
                    this_center_map[y][x][0] - this.point_size,
                    this_center_map[y][x][1] - this.point_size,
                    this.point_size * 2, this.point_size * 2,
                )
                this.resetStyle()
                this.detaildiv.innerHTML = this_annotations[this_anno_idx_map[y][x]]['label']
                this.detaildiv.style.width = 12 + 12 * this.detaildiv.innerHTML.length + 'px'
                this.detaildiv.style.display = ''
                this.detaildiv.style.left = e.clientX + 16 + 'px'
                this.detaildiv.style.top = e.clientY + 16 + "px"
            } else {
                this.detaildiv.style.display = 'none'
            }
            //
            if (this.draw_polygon_mode) {
                if (this_annotation['path'].length != 0) {
                    this.drawCurAnno()
                    this.lineTo(offset_x, offset_y)
                    this.stroke()
                }
                if (
                    this_annotation['path'].length > 3
                    && Math.abs(offset_x - this_annotation['path'][0]['x']) < this.point_size
                    && Math.abs(offset_y - this_annotation['path'][0]['y']) < this.point_size
                ) {
                    this.fillRect(
                        this_annotation['path'][0]['x'] - this.point_size,
                        this_annotation['path'][0]['y'] - this.point_size,
                        this.point_size * 2,
                        this.point_size * 2,
                    )
                }
            } else if (this.insert_mode) {
                if (this_anno_idx_map[y][x] != -1 && !this_is_bbox_map[y][x] && !this_is_line_map[y][x]) {
                    let annotation = this_annotations[this_anno_idx_map[y][x]]
                    let path_idx = (this_path_idx_map[y][x] - 1 + annotation['path'].length) % annotation['path'].length
                    this.setStyle(annotation['label'])
                    this.fillRect(
                        annotation['path'][path_idx]['x'] - this.point_size,
                        annotation['path'][path_idx]['y'] - this.point_size,
                        this.point_size * 2, this.point_size * 2,
                    )
                    this.resetStyle()
                }
            } else if ((this.region_mode || this.draw_bbox_mode) && this.selecting) {
                let xmin = min(this.bbox['xmin'], offset_x)
                let xmax = max(this.bbox['xmin'], offset_x)
                let ymin = min(this.bbox['ymin'], offset_y)
                let ymax = max(this.bbox['ymin'], offset_y)
                this.strokeRect(
                    xmin, ymin, xmax - xmin, ymax - ymin,
                )
            } else if (this.draw_line_mode && this.selecting) {
                let prev_line_width = this_context.lineWidth
                this_context.lineWidth = prev_line_width + 1
                this.moveTo(this.line['x0'], this.line['y0'])
                this.lineTo(offset_x, offset_y)
                this.stroke()
                this_context.lineWidth = prev_line_width
            } else {
                if (this.drag_mode) {
                    if (this.drag_bbox) {
                        if (this.point_idx == 0) {
                            // 左上角
                            this_annotation['bbox']['xmin'] = offset_x
                            this_annotation['bbox']['ymin'] = offset_y
                        } else if (this.point_idx == 1) {
                            // 右上角
                            this_annotation['bbox']['xmax'] = offset_x
                            this_annotation['bbox']['ymin'] = offset_y
                        } else if (this.point_idx == 2) {
                            // 左下角
                            this_annotation['bbox']['xmin'] = offset_x
                            this_annotation['bbox']['ymax'] = offset_y
                        } else if (this.point_idx == 3) {
                            // 右下角
                            this_annotation['bbox']['xmax'] = offset_x
                            this_annotation['bbox']['ymax'] = offset_y
                        }
                    } else if (this.drag_line) { 
                        if (this.point_idx == 0) {
                            this_annotation['line']['x0'] = offset_x
                            this_annotation['line']['y0'] = offset_y
                        } else {
                            this_annotation['line']['x1'] = offset_x
                            this_annotation['line']['y1'] = offset_y
                        }
                    } else {
                        this_annotation['path'][this.point_idx]['x'] = offset_x
                        this_annotation['path'][this.point_idx]['y'] = offset_y
                    }
                }
            }
        },

        mouseDown(e: any) {
            console.log(e.button, e.buttons)
            if (
                e.offsetX < 0 || e.offsetX >= this.region_info['width'] * this.region_info['ratio']
                || e.offsetY < 0 || e.offsetY >= this.region_info['height'] * this.region_info['ratio']
            ) {
                // 点击位置在图像区域外
                console.log('out')
                return
            }
            let offset_x = e.offsetX / this.region_info['ratio'] + this.region_info['xmin']
            let offset_y = e.offsetY / this.region_info['ratio'] + this.region_info['ymin']
            let x = parseInt(offset_x + '', 10)
            let y = parseInt(offset_y + '', 10)
            if (x < 0 || y < 0 || x >= this.image.width || y >= this.image.height)
                return
            //
            if (this.draw_polygon_mode) {
                if (
                    e.button == 2
                    || (
                        this_annotation['path'].length > 2
                        && Math.abs(offset_x - this_annotation['path'][0]['x']) < this.point_size
                        && Math.abs(offset_y - this_annotation['path'][0]['y']) < this.point_size
                    )
                ) {
                    this.operate('c')
                    // 添加标注时，再次点击第一个点
                    // this.maskAllModes()
                    // this.drawCurAnno()
                    // this.closePath()
                    // this.stroke()
                    // let anno_label = prompt('请输入当前标注的标签[1, 2, 3, 4, 5, 6]')
                    // this_annotation['label'] = anno_label
                    // this_annotations.push(this_annotation)
                    // this_annotation = {}
                } else if (this_annotation['path'].length == 0) {
                    // 第一个点
                    this_annotation['path'].push({
                        'x': offset_x,
                        'y': offset_y,
                    })
                    this.moveTo(offset_x, offset_y)
                } else {
                    // 点击第二、三...个点
                    this_annotation['path'].push({
                        'x': offset_x,
                        'y': offset_y,
                    })
                    this.drawCurAnno()
                    this.lineTo(offset_x, offset_y)
                    this.stroke()
                }
            } else if (this.delete_mode) {
                // 删除标注
                if (this_anno_idx_map[y][x] != -1) {
                    console.log('delete')
                    this_annotations.splice(this_anno_idx_map[y][x], 1)
                    this.maskAllModes()
                    this_annotation = {}
                    this.drawImageAnno()
                }
            } else if (this.insert_mode) {
                if (this_anno_idx_map[y][x] != -1 && !this_is_bbox_map[y][x] && !this_is_line_map[y][x]) {
                    console.log('adding point')
                    this.maskAllModes()
                    this.drag_mode = true
                    this_annotation = this_annotations[this_anno_idx_map[y][x]]
                    this.point_idx = this_path_idx_map[y][x]
                    this_annotation['path'].splice(this.point_idx, 0, {'x': x, 'y': y})
                }
            } else if (this.pop_mode) {
                if (this_anno_idx_map[y][x] != -1 && !this_is_bbox_map[y][x] && !this_is_line_map[y][x]) {
                    let annotation = this_annotations[this_anno_idx_map[y][x]]
                    if (annotation['path'].length > 3) {
                        annotation['path'].splice(this_path_idx_map[y][x], 1)
                    }
                    this.maskAllModes()
                }
            } else if (this.region_mode || this.draw_bbox_mode) {
                if (!this.selecting) {
                    // 选择矩形的第一个点
                    this.selecting = true
                    this.bbox['xmin'] = offset_x
                    this.bbox['ymin'] = offset_y
                } else {
                    // 选择矩形的第二个点
                    let xmin = min(this.bbox['xmin'], offset_x)
                    let xmax = max(this.bbox['xmin'], offset_x)
                    let ymin = min(this.bbox['ymin'], offset_y)
                    let ymax = max(this.bbox['ymin'], offset_y)
                    if (this.region_mode) {
                        this.region_info['xmin'] = xmin
                        this.region_info['ymin'] = ymin
                        this.region_info['width'] = xmax - xmin
                        this.region_info['height'] = ymax - ymin
                        this.region_info['ratio'] = min(
                            this_bg.width / this.region_info['width'],
                            this_bg.height / this.region_info['height'],
                        )
                        this.point_size = this.base_point_size / this.region_info['ratio']
                        this.maskAllModes()
                    } else if (this.draw_bbox_mode) {
                        this.bbox['xmin'] = xmin
                        this.bbox['ymin'] = ymin
                        this.bbox['xmax'] = xmax
                        this.bbox['ymax'] = ymax
                        if (!this.prompt_mode) {
                            this.operate('c')
                        } else {
                            this.prompt['bbox'] = this.bbox
                            this.draw_bbox_mode = false
                            this.bbox = {}
                            this.SAMPredict()
                        }
                    }
                }
            } else if (this.draw_line_mode) {
                if (!this.selecting) {
                    // 选择线段第一个点
                    this.selecting = true
                    this.line['x0'] = offset_x
                    this.line['y0'] = offset_y
                } else {
                    // 选择线段第二个点
                    this.line['x1'] = offset_x
                    this.line['y1'] = offset_y
                    this.operate('c')
                }
            } else if (this.prompt_mode) {
                // SAM点击模式
                this.prompt['points'].push({
                    'x': offset_x,
                    'y': offset_y,
                })
                if (e.button == 0) {
                    this.prompt['labels'].push(1)
                } else {
                    this.prompt['labels'].push(0)
                }
                this.SAMPredict()
            } else {
                if (!this.drag_mode) {
                    // 选择要拖拽的点
                    if (this_anno_idx_map[y][x] != -1) {
                        console.log('drag_mode')
                        this.drag_mode = true
                        this.drag_index = this_anno_idx_map[y][x]
                        this_annotation = this_annotations[this.drag_index]
                        this.drag_bbox = this_is_bbox_map[y][x]
                        this.drag_line = this_is_line_map[y][x]
                        this.point_idx = this_path_idx_map[y][x]
                    }
                } else {
                    // 选择要拖拽到的位置
                    this.drag_mode = false
                    this_annotation['path'][this.point_idx]['x'] = offset_x
                    this_annotation['path'][this.point_idx]['y'] = offset_y
                }
            }
            this.reinitializeMaps()
        },

        mouseUp(e: any) {
        },

        operate(key: string) {
            console.log('key', key)
            switch (key) {
                case 'a':   // 'a' --> 'append'
                    // 添加标注
                    if (this.checkAllModes()) {
                        this.draw_polygon_mode = true
                        this_annotation = {'path': []}
                        this.status.innerHTML = this.descriptions['draw_polygon_mode']
                    }
                    break
                case 'b':   // 'b' --> 'bbox'
                    // 添加多边形标注
                    if (this.prompt_mode || this.checkAllModes()) {
                        this.draw_bbox_mode = true
                        this.bbox = {}
                        this.status.innerHTML = this.descriptions['draw_bbox_mode']
                    }
                    break
                case 'l':   // 'l' --> 'line'
                    // 添加线段标注
                    if (this.checkAllModes()) {
                        this.draw_line_mode = true
                        this.line = {}  // x0, y0, x1, y1
                        this.status.innerHTML = this.descriptions['draw_line_mode']
                    }
                    break;
                case 'u':   // 'u' --> 'undo'
                    // 添加标注时，回退一步
                    if (this.draw_polygon_mode) {
                        if (this_annotation['path'].length > 0) {
                            this_annotation['path'].pop()
                        }
                        this.drawImageAnno()
                        this.drawCurAnno()
                        this.stroke()
                    } else if (this.prompt_mode) {
                        if (this.prompt['points'].length > 0) {
                            this.prompt['points'].pop()
                            this.prompt['labels'].pop()
                        }
                        this.SAMPredict()
                    }
                    break
                case 'c':   // 'c' --> complete
                    // 完成标注
                    if (
                        this.draw_polygon_mode
                        && 'path' in this_annotation
                        && this_annotation['path'].length > 2
                    ) {
                        this.maskAllModes()
                        this.drawCurAnno()
                        this.closePath()
                        this.stroke()
                        let anno_label = prompt('请输入当前标注的标签[1, 2, 3, 4, 5, 6]')
                        this_annotation['label'] = anno_label
                        this_annotations.push(this_annotation)
                    } else if (this.prompt_mode) {
                        // 这里隐含了draw_bbox_mode = true的情况
                        this.maskAllModes()
                        let anno_label = prompt('请输入当前标注的标签[1, 2, 3, 4, 5, 6]')
                        for (let i = 0; i < this_sam_annotations.length; i++) {
                            // this_sam_annotations[i]['label'] = anno_label
                            let valid_sub_path = this.checkPath(this_sam_annotations[i]['path'])['valid_sub_path']
                            this_annotations.push({'label': anno_label, 'path': valid_sub_path})
                        }
                    } else if (this.draw_bbox_mode) {
                        this.maskAllModes()
                        let anno_label = prompt('请输入当前标注的标签[1, 2, 3, 4, 5, 6]')
                        this_annotations.push({'label': anno_label, 'bbox': this.bbox})
                    } else if (this.draw_line_mode) {
                        this.maskAllModes()
                        let anno_label = prompt('请输入当前标注的标签[1, 2, 3, 4, 5, 6]')
                        this_annotations.push({'label': anno_label, 'line': this.line})
                    }
                    this_annotation = {}
                    this.bbox = {}
                    this.line = {}
                    this.prompt['points'] = []
                    this.prompt['labels'] = []
                    this.prompt['bbox'] = []
                    this_sam_annotations = []
                    this.drawImageAnno()
                    break
                case 'q':   // 'q' --> 'quit'
                    if (this.prompt_mode || this.draw_polygon_mode || this.draw_bbox_mode || this.draw_line_mode) {
                        // 退出标注
                        this.maskAllModes()
                        this_annotation = {}
                        this.bbox = {}
                        this.line = {}
                        this.prompt['points'] = []
                        this.prompt['labels'] = []
                        this.prompt['bbox'] = []
                        this_sam_annotations = []
                        this.drawImageAnno()
                    } else {
                        // 重置
                        this.maskAllModes()
                        this.resetRegion()
                        this.drawImageAnno()
                    }
                    break
                case 'm':
                    // 使用SAM做图像分割
                    if (this.checkAllModes()) {
                        this.prompt_mode = true
                        this.status.innerHTML = this.descriptions['prompt_mode']
                        this_sam_annotations = []
                    }
                    break
                default:
            }
            if (!this.checkAllModes()) {
                return
            }
            switch (key) {
                case 'i':   // 'i' --> 'insert'
                    // 插入一个顶点
                    if (this.checkAllModes()) {
                        this.insert_mode = true
                        this.status.innerHTML = this.descriptions['insert_mode']
                    }
                    break
                case 'p':   // 'p' --> 'pop'
                    // 删除一个顶点
                    if (this.checkAllModes()) {
                        this.pop_mode = true
                        this.status.innerHTML = this.descriptions['pop_mode']
                    }
                    break
                case 'd':   // 'd' --> 'delete'
                    // 删除指定标注
                    if (this.checkAllModes()) {
                        this.delete_mode = true
                        this.status.innerHTML = this.descriptions['delete_mode']
                    }
                    this.drawImageAnno()
                    break
                case 's':
                    // 保存标注编辑
                    axios
                        .post(
                            backend_address + '/save_anno',
                            {
                                'token': token,
                                'collection_name': collection_name,
                                'image_name': this.image_name,
                                'anno': this_annotations,
                            },
                        )
                        .then(response => {
                            alert('保存成功')
                        })
                        .catch(error => {
                            alert(error)
                        })
                        .finally(() => {})
                    break
                case 'r':
                    // 放大区域
                    this.region_mode = true
                    this.status.innerHTML = this.descriptions['region_mode']
                    this.resetRegion()
                    this.drawImageAnno()
                    break
                case 'clear_anno':
                    // 删除所有标注
                    this_annotation = {}
                    this_annotations = []
                    this.drawImageAnno()
                    break
                case 'dilate_anno':
                    this.dilateAnno()
                    break
                case 'erode_anno':
                    this.erodeAnno()
                    break
                case 'rename':
                    // 重命名
                    this.renameImage()
                    break
                case 'remove':
                    // 删除图片
                    this.removeImage()
                    break
                case 'download_anno':
                    // 下载标注文件，png格式
                    this.downloadAnno()
                    break
                case 'download_fg':
                    // 下载抠图，png格式
                    this.downloadFG()
                    break
                case 'next':
                    // 切换到下一张图片
                    this.nextImage()
                    break
                case 'prev':
                    // 切换到上一张图片
                    this.prevImage()
                    break
                case 'calc_volume':
                    // 计算标注面积
                    this.calcVolume()
                    break
                case 'instruct':
                    // 说明书
                    this.instruct()
                    break
                case 'z':
                    // debug用
                    console.log('Test key "z"')
                    axios
                        .post(
                            backend_address + '/test', {'token': token, 'message': 'hello'},
                        )
                        .then(response => {
                            console.log(response)
                        })
                        .catch(error => {
                            console.log(error)
                        })
                        .finally(() => {})
                    break
                default:
            }
            return
        },

        keyDown(e: any) {
            this.operate(e.key)
        },

        scrollFunc(e: any)  {
            e.preventDefault()
            e = e || window.event
            let offset_x = e.offsetX / this.region_info['ratio'] + this.region_info['xmin']
            let offset_y = e.offsetY / this.region_info['ratio'] + this.region_info['ymin']
            let x = parseInt(offset_x + '', 10)
            let y = parseInt(offset_y + '', 10)
            if (
                x >= 0 && y >= 0 && x < this_bg.width && y < this_bg.height
            ) {
                if (e.wheelDelta) {
                    if (e.wheelDelta > 0) {
                        this.nextImage()
                    } else if (e.wheelDelta < 0) {
                        this.prevImage()
                    }
                }
            }
        },

        // Initialize context and maps
        initialize() {
            // These two lines will be deleted in the future
            // this_bg.height = 1024
            // this_bg.width = 1980
            // Initialize region_info
            // this.resetRegion()
            // Initialize maps
            this_anno_idx_map = []
            this_is_bbox_map = []
            this_is_line_map = []
            this_path_idx_map = []
            this_center_map = []
            for (let i = 0; i < this.image.height; i++) {
                let anno_idx_arr = []
                let is_bbox_arr = []
                let is_line_arr = []
                let path_idx_arr = []
                let center_arr = []
                for (let j = 0; j < this.image.width; j++) {
                    anno_idx_arr.push(-1)
                    is_bbox_arr.push(0)
                    is_line_arr.push(0)
                    path_idx_arr.push(-1)
                    center_arr.push([0, 0])
                }
                this_anno_idx_map.push(anno_idx_arr)
                this_is_bbox_map.push(is_bbox_arr)
                this_is_line_map.push(is_line_arr)
                this_path_idx_map.push(path_idx_arr)
                this_center_map.push(center_arr)
            }
            const self = this
            //
            this.upload_button = document.getElementById('upload')
            this.upload_button.onchange = function() {
                let image_name = this.files[0]['name']
                let reader = new FileReader()
                reader.readAsDataURL(this.files[0])
                reader.onload = function(e) {
                    axios
                        .post(
                            backend_address + '/upload_image', {
                                'token': token,
                                'collection_name': collection_name,
                                'image': this.result,
                                'image_name': image_name,
                            },
                        )
                        .then(response => {
                            console.log('uploaded')
                            self.readCollectionList().then(async () => {
                                console.log('finish readCollectionList')
                                // await nextTick()
                                // 获取HTMLSelectElement元素
                                let image_selecter = document.getElementById('image_selecter') as HTMLSelectElement
                                // 遍历每个选项
                                for (let i = 0; i < image_selecter.options.length; i++) {
                                    // 检查每个选项的文本是否为'233'
                                    console.log(image_selecter.options[i].text + ' vs ' + image_name)
                                    if (image_selecter.options[i].text === image_name) {
                                        // 将当前选项设置为该选项
                                        image_selecter.selectedIndex = i
                                        break // 找到后跳出循环
                                    }
                                }
                                self.onImageChange('')
                            })
                            // alert('上传成功，请刷新网页')
                        })
                        .catch(error => {
                            console.log(error)
                        })
                        .finally(() => {})
                }
            }
            this.upload_json_button = document.getElementById('upload_json')
            this.upload_json_button.onchange = function() {
                let json_name = this.files[0]['name']
                let reader = new FileReader()
                reader.readAsText(this.files[0])
                reader.onload = function(e) {
                    axios
                        .post(backend_address + '/upload_json', {
                            'token': token,
                            'collection_name': collection_name,
                            'content': this.result,
                            'json_name': json_name,
                        })
                        .then(response => {
                            self.readCollectionList()
                            // alert('上传成功')
                        })
                        .catch(error => {
                            console.log(error)
                        })
                        .finally(() => {})
                }
            }
        },

        reinitializeMaps() {
            for (let i = 0; i < this.image.height; i++) {
                for (let j = 0; j < this.image.width; j++) {
                    this_anno_idx_map[i][j] = -1
                    this_is_bbox_map[i][j] = 0
                    this_is_line_map[i][j] = 0
                    this_path_idx_map[i][j] = -1
                    this_center_map[i][j] = [0, 0]
                }
            }
        },

        fillMap(x: number, y: number, size: number, anno_idx: number, is_rect: boolean, is_line: boolean, path_idx: number) {
            let min_i = Math.max(parseInt((y - size / 2) + '', 10), 0)
            let max_i = Math.min(parseInt((y + size / 2) + '', 10), this.image.height)
            let min_j = Math.max(parseInt((x - size / 2) + '', 10), 0)
            let max_j = Math.min(parseInt((x + size / 2) + '', 10), this.image.width)
            for (let i = min_i; i < max_i; i++) {
                for (let j = min_j; j < max_j; j++) {
                    this_anno_idx_map[i][j] = anno_idx
                    this_is_bbox_map[i][j] = is_rect
                    this_is_line_map[i][j] = is_line
                    this_path_idx_map[i][j] = path_idx
                    this_center_map[i][j] = [x, y]
                }
            }
        },

        renameImage() {
            let new_name = prompt('当前文件 <' + this.image_name + '>' + '请输入一个新的文件名（不包含后缀）')
            if (new_name == '') {
                return
            }
            axios
                .post(
                    backend_address + '/rename_image',
                    {
                        'token': token,
                        'collection_name': collection_name,
                        'origin_name': this.image_name,
                        'new_name': new_name,
                    },
                )
                .then(response => {
                    console.log('new_name is', response.data)
                })
                .catch(error => {
                    console.log('rename image error')
                })
                .finally(() => {})
            this.readCollectionList()
        },

        removeImage() {
            let res = confirm('当前文件 <' + this.image_name + '> 删除是不可恢复的，你确认要删除吗？')
            if (!res) {
                return
            }
            axios
                .post(
                    backend_address + '/remove_image',
                    {
                        'token': token,
                        'collection_name': collection_name,
                        'image_name': this.image_name,
                    },
                )
                .then(response => {
                })
                .catch(error => {
                    console.log('remove image error')
                })
                .finally(() => {})
            this.readCollectionList()
        },

        downloadAnno() {
            axios
                .post(
                    backend_address + '/get_anno_png',
                    {
                        'token': token,
                        'collection_name': collection_name,
                        'image_name': this.image_name,
                    },
                )
                .then(response => {
                    let encryptedBytes = response.data
                    console.log(encryptedBytes)
                    let link = document.createElement('a')
                    link.href = 'data:image/png;base64,' + encryptedBytes.toString('base64')
                    link.download = 'anno.png'
                    document.body.appendChild(link)
                    link.click()
                    document.body.removeChild(link)
                })
                .catch(error => {
                    console.log(error)
                })
                .finally(() => {})
        },

        downloadFG() {
            axios
                .post(
                    backend_address + '/get_fg_png',
                    {
                        'token': token,
                        'collection_name': collection_name,
                        'image_name': this.image_name,
                    },
                )
                .then(response => {
                    let encryptedBytes = response.data
                    console.log(encryptedBytes)
                    let link = document.createElement('a')
                    link.href = 'data:image/png;base64,' + encryptedBytes.toString('base64')
                    link.download = 'anno.png'
                    document.body.appendChild(link)
                    link.click()
                    document.body.removeChild(link)
                })
                .catch(error => {
                    console.log(error)
                })
                .finally(() => {})
        },


        instruct() {
            alert(
                '多边形标注：鼠标左键多次点击绘制多边形\n'
                + '框标注：鼠标左键两次点击绘制矩形框\n'
                + 'SAM标注：使用SAM进行半自动标注，鼠标左键选择前景点，右键选择背景点\n'
                + '回退一步：在绘制多边形过程中，点击该按钮回退一步；在SAM标注模式中，点击该按钮撤回一个点\n'
                + '完成标注：在绘制多边形过程（或在SAM标注模式）中，点击该按钮完成绘制\n'
                + '退出标注：在添加标注任意一步（或在SAM标注模式）中，点击该按钮退出标注进程\n'
                + '删除标注：进入删除标注的状态，点击任意顶点删除对应标注\n'
                + '增加顶点：进入增加顶点的状态，点击任意多边形顶点以增加顶点\n'
                + '删除顶点：进入删除顶点的状态，点击任意多边形顶点以删除顶点\n'
                + '保存标注：保存当前标注编辑结果\n'
                + '下载标注：下载当前标注的png结果\n'
                + '放大：放大局部区域，指定区域的方式是绘制一个矩形框\n'
                + '重置：重置所有状态和放大模式\n'
                + '上传图片：上传一张图片到当前目录下，格式为png、jpg、jpeg\n'
                + '上传标注：上传一个标注文件，格式为json\n'
                + '重命名：重命名当前图片\n'
                + '删除图片：删除当前图片\n'
                // + '分辨率比例：修改图片的分辨率，有0.25 0.5 1.0 2.0 4.0五个选项\n'
                + '选择图片：滑动滑块可以快速选择当前目录下的图片\n'
                + '上一张图：跳转到上一张图\n'
                + '下一张图：跳转到下一张图\n',
            )
        },


        addListenerForButtons() {
            (document.getElementById('add') as HTMLButtonElement).addEventListener('click', () => this.operate('a'));
            (document.getElementById('bbox') as HTMLButtonElement).addEventListener('click', () => this.operate('b'));
            (document.getElementById('close') as HTMLButtonElement).addEventListener('click', () => this.operate('c'));
            (document.getElementById('delete') as HTMLButtonElement).addEventListener('click', () => this.operate('d'));
            (document.getElementById('insert') as HTMLButtonElement).addEventListener('click', () => this.operate('i'));
            (document.getElementById('line') as HTMLButtonElement).addEventListener('click', () => this.operate('l'));
            (document.getElementById('pop') as HTMLButtonElement).addEventListener('click', () => this.operate('p'));
            (document.getElementById('quit') as HTMLButtonElement).addEventListener('click', () => this.operate('q'));
            (document.getElementById('reset') as HTMLButtonElement).addEventListener('click', () => {this.operate('q')});
            (document.getElementById('region') as HTMLButtonElement).addEventListener('click', () => this.operate('r'));
            (document.getElementById('save') as HTMLButtonElement).addEventListener('click', () => this.operate('s'));
            (document.getElementById('undo') as HTMLButtonElement).addEventListener('click', () => this.operate('u'));
            //
            (document.getElementById('rename') as HTMLButtonElement).addEventListener('click', () => {this.operate('rename')});
            (document.getElementById('remove') as HTMLButtonElement).addEventListener('click', () => {this.operate('remove')});
            (document.getElementById('download_anno') as HTMLButtonElement).addEventListener('click', () => {this.downloadAnno()});
            (document.getElementById('download_fg') as HTMLButtonElement).addEventListener('click', () => {this.downloadFG()});
            //
            (document.getElementById('next') as HTMLButtonElement).addEventListener('click', () => {this.operate('next')});
            (document.getElementById('prev') as HTMLButtonElement).addEventListener('click', () => {this.operate('prev')});
            (document.getElementById('calc_volume') as HTMLButtonElement).addEventListener('click', () => {this.operate('calc_volume')});
            (document.getElementById('instruct') as HTMLButtonElement).addEventListener('click', () => {this.operate('instruct')});
            (document.getElementById('sam_pred') as HTMLButtonElement).addEventListener('click', () => {this.operate('m')});
            (document.getElementById('clear_anno') as HTMLButtonElement).addEventListener('click', () => {this.operate('clear_anno')});
            (document.getElementById('dilate_anno') as HTMLButtonElement).addEventListener('click', () => {this.operate('dilate_anno')});
            (document.getElementById('erode_anno') as HTMLButtonElement).addEventListener('click', () => {this.operate('erode_anno')})
        },
    },

    mounted() {
        token = window.location.search.substring(1)
        // Initialize components
        this_bg = document.getElementById('bg') as HTMLCanvasElement
        this_context = this_bg.getContext('2d') as CanvasRenderingContext2D
        this_bg.onmousemove = throttle(this.mouseMove.bind(this), 16)
        this_bg.onmousedown = this.mouseDown.bind(this)
        this_bg.onmouseup = this.mouseUp.bind(this)
        // collection selecter
        this.collection_selecter = document.getElementById('collection_selecter') as HTMLSelectElement
        this.collection_selecter.addEventListener('change', this.onCollectionChange)
        // image selecter
        this.image_selecter = document.getElementById('image_selecter') as HTMLSelectElement
        this.image_selecter.addEventListener('change', this.onImageChange)
        // sam_selecter
        this.sam_selecter = document.getElementById('sam_selecter') as HTMLSelectElement
        // this.sam_selecter.addEventListener('change', this.onSAMChange)
        this.sam_selecter.selectedIndex = 1
        // status div
        this.status = document.getElementById('status') as HTMLDivElement
        this.status.innerHTML = this.descriptions['drag_mode']
        window.addEventListener('keydown', this.keyDown)
        this.addListenerForButtons()
        // scroll
        window.addEventListener('mousewheel', this.scrollFunc, { passive: false })
        // debug div
        this.debugdiv = document.getElementById('debug_block') as HTMLDivElement
        // input text
        this.dilate_max_objs = document.getElementById('dilate_max_objs')
        this.dilate_thres = document.getElementById('dilate_thres')
        // range
        this.comp_force_range = document.getElementById('comp_force')
        this.show_anno = document.getElementById('show_anno')
        this.show_anno.onchange = this.drawImageAnno
        // this.adjust_force_range = document.getElementById('adjust_force')
        this.select_image_range = document.getElementById('select_image')
        this.select_image_range.addEventListener('change', this.onSelectImageChange)
        this.select_image_range.addEventListener('mousemove', this.onSelectImageMove)
        this.select_image_range.addEventListener('mouseout', this.onSelectImageOut)
        // Detail Div
        this.detaildiv = document.createElement('div')
        this.detaildiv.align = 'center'
        this.detaildiv.id = 'detail'
        this.detaildiv.style.backgroundColor = '#ffffff'
        this.detaildiv.style.height = '32px'
        this.detaildiv.style.border = '1px solid #666666'
        this.detaildiv.style.position = 'absolute'
        document.body.appendChild(this.detaildiv)
        // 更新标签颜色表
        for (let i = 0; i < this.style_list.length; i++) {
            let element = document.getElementById('color_' + (i + 1))
            if (element instanceof HTMLElement) {
                element.style.backgroundColor = this.style_list[i]
            }
        }
        // Initialize gloabl variables
        this_annotation = {}
        this_annotations = []
        this.bbox = {}
        this.line = {}
        // Read image
        // this.image_name = 'yousa.png'
        // this.image_list = [{'image_name': 'yousa.png'}]
        // this.readImage('yousa.png')
        this.readCollectionList()
        // window.setInterval(() => {
        //     this.readImage('yousa.png')
        // }, 3000);
    },
}
</script>

<style scoped>


.up_selecter {
    width: 200px;
    height: 36px;
    margin: 8px;
    padding: 4px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 16px;
    color: #333;
}
/* 鼠标悬停样式 */
.up_selecter:hover {
    border-color: #999;
}
/* 选中样式 */
.up_selecter:focus {
    outline: none;
    border-color: #666;
    box-shadow: 0 0 5px rgba(0, 0, 0, 0.3);
}

/* 最上方的说明文字 */
#status {
    margin:8px auto;
    font-size:24px;
    /* color:#ff9999; */
    border-radius:16px;
}

.button {
    width:100px;
    height:36px;
    text-align:center;
    line-height:100%;
    padding:0.3em;
    font: Arial,sans-serif bold;
    font-size: 14px;
    font-style:normal;
    text-decoration:none;
    margin:6px;
    vertical-align:text-bottom;
    zoom:1;
    outline:none;
    font-size-adjust:none;
    font-stretch:normal;
    border-radius:12px;
    border: 1px solid #6C8EBF;
    color:#000000;
    background-repeat:repeat;
    background-size:auto;
    background-origin:padding-box;
    background-clip:padding-box;
    background-color:#DAE8FC;
}
.button:hover {
    background: #6C8EBF;
}

.small_button {
    width:90px;
    height:36px;
    text-align:center;
    line-height:100%;
    padding:0.3em;
    font: Arial,sans-serif bold;
    font-size: 14px;
    font-style:normal;
    text-decoration:none;
    margin:6px;
    vertical-align:text-bottom;
    zoom:1;
    outline:none;
    font-size-adjust:none;
    font-stretch:normal;
    border-radius:12px;
    border: 1px solid #6C8EBF;
    color:#000000;
    background-repeat:repeat;
    background-size:auto;
    background-origin:padding-box;
    background-clip:padding-box;
    background-color:#DAE8FC;
}
.small_button:hover {
    background: #6C8EBF;
}

.button_box {
    border-radius:12px;
    border: 1px solid #6C8EBF;
}

.text {
    font-size:20px;
}
.small_text {
    font-size:16px;
}

/* 基本样式 */
.select_range {
    width: 160px; /* 设置宽度 */
    height: 20px; /* 设置高度 */
    margin: 8px; /* 设置外边距 */
  }
  
#comp_force {
    width: 80px;
}

#dilate_thres {
    width: 80px
}

/* 自定义滑块样式 */
.select_range::-webkit-slider-thumb {
    -webkit-appearance: none; /* 隐藏默认的滑块样式 */
    width: 20px; /* 滑块宽度 */
    height: 20px; /* 滑块高度 */
    background: #DAE8FC; /* 滑块背景颜色 */
    border-radius: 50%; /* 滑块圆角 */
    cursor: pointer; /* 鼠标指针样式 */
}

/* 自定义滑块在不同状态下的样式 */
.select_range:hover::-webkit-slider-thumb {
    background: #DAE8FC; /* 鼠标悬停时的滑块颜色 */
}

.select_range:focus::-webkit-slider-thumb {
    background: #DAE8FC; /* 选中状态的滑块颜色 */
}

label, input, select{
  vertical-align: middle;
}

/* 隐藏默认的复选框样式 */
.mui-switch {
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
    width: 32px;
    height: 20px;
    border-radius: 20px;
    position: relative;
    cursor: pointer;
    outline: none;
    background-color: #ccc;
    transition: background-color 0.3s;
  }
  
  /* 自定义复选框的选中状态样式 */
  .mui-switch:checked {
    background-color: #6C8EBF;
  }
  
  /* 复选框的滑块样式 */
  .mui-switch::after {
    content: '';
    position: absolute;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background-color: white;
    top: 50%;
    left: 4px;
    transform: translateY(-50%);
    transition: transform 0.3s, background-color 0.3s;
  }
  
  /* 复选框选中状态下的滑块样式 */
  .mui-switch:checked::after {
    left: calc(100% - 20px);
    transform: translateY(-50%);
  }
.container {
    display: flex; /* 使用 Flexbox 布局 */
    align-items: center; /* 垂直居中对齐 */
    justify-content: center;
}

.color-table {
    display: flex;
}
  
.color {
    width: 50px;
    height: 50px;
    text-align: center;
    line-height: 50px;
    font-weight: bold;
    font-size: 24px;
    margin: 5px;
    color: black;
}

#dilate_max_objs {
    width: 50px;
    text-align: center;
    border: 1px solid #DAE8FC;
    border-radius:12px;
    font: 16px Arial,sans-serif bold;
    font-style: normal;
}

</style>
