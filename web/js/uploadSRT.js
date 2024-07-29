import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

function srtUpload(node, inputName, inputData, app) {
  const srtWidget = node.widgets.find((w) => w.name === "srt");
  let uploadWidget;
  /* 
    A method that returns the required style for the html 
    */
  var default_value = srtWidget.value;
  Object.defineProperty(srtWidget, "value", {
    set: function (value) {
      this._real_value = value;
    },

    get: function () {
      let value = "";
      if (this._real_value) {
        value = this._real_value;
      } else {
        return default_value;
      }

      if (value.filename) {
        let real_value = value;
        value = "";
        if (real_value.subfolder) {
          value = real_value.subfolder + "/";
        }

        value += real_value.filename;

        if (real_value.type && real_value.type !== "input")
          value += ` [${real_value.type}]`;
      }
      return value;
    },
  });
  async function uploadFile(file, updateNode, pasted = false) {
    try {
      // Wrap file in formdata so it includes filename
      const body = new FormData();
      body.append("image", file);
      if (pasted) body.append("subfolder", "pasted");
      const resp = await api.fetchApi("/upload/image", {
        method: "POST",
        body,
      });

      if (resp.status === 200) {
        const data = await resp.json();
        // Add the file to the dropdown list and update the widget value
        let path = data.name;
        if (data.subfolder) path = data.subfolder + "/" + path;

        if (!srtWidget.options.values.includes(path)) {
          srtWidget.options.values.push(path);
        }

        if (updateNode) {
          srtWidget.value = path;
        }
      } else {
        alert(resp.status + " - " + resp.statusText);
      }
    } catch (error) {
      alert(error);
    }
  }

  const fileInput = document.createElement("input");
  Object.assign(fileInput, {
    type: "file",
    accept: "file/srt,file/txt",
    style: "display: none",
    onchange: async () => {
      if (fileInput.files.length) {
        await uploadFile(fileInput.files[0], true);
      }
    },
  });
  document.body.append(fileInput);

  // Create the button widget for selecting the files
  uploadWidget = node.addWidget(
    "button",
    "choose srt file to upload",
    "Audio",
    () => {
      fileInput.click();
    },
  );

  uploadWidget.serialize = false;

  const cb = node.callback;
  srtWidget.callback = function () {
    if (cb) {
      return cb.apply(this, arguments);
    }
  };

  return { widget: uploadWidget };
}

ComfyWidgets.SRTPLOAD = srtUpload;

const setupDynamicConnections = (nodeType, prefix, inputType, opts) => {
  /** @type {{link?:LLink, ioSlot?:INodeInputSlot | INodeOutputSlot}} */
  const options = opts || {};
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  const inputList = typeof inputType === "object";

  nodeType.prototype.onNodeCreated = function () {
    const r = onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
    const input = options.nameArray ? options.nameArray[0] : `${prefix}_1`;

    this.addProperty("dynamicInputsIndex", this.inputs.length);

    this.addInput(input, inputList ? "*" : inputType);
    return r;
  };

  const onConnectionsChange = nodeType.prototype.onConnectionsChange;
  /**
   * @param {OnConnectionsChangeParams} args
   */
  nodeType.prototype.onConnectionsChange = function (...args) {
    const [type, slotIndex, isConnected, link, ioSlot] = args;

    options.link = link;
    options.ioSlot = ioSlot;
    const r = onConnectionsChange
      ? onConnectionsChange.apply(this, [
          type,
          slotIndex,
          isConnected,
          link,
          ioSlot,
        ])
      : undefined;
    options.DEBUG = {
      node: this,
      type,
      slotIndex,
      isConnected,
      link,
      ioSlot,
    };

    dynamic_connection(
      this,
      slotIndex,
      isConnected,
      `${prefix}_`,
      inputType,
      options,
    );
    return r;
  };
};

const nodesFromLink = (node, link) => {
  const fromNode = app.graph.getNodeById(link.origin_id);
  const toNode = app.graph.getNodeById(link.target_id);

  let tp = "error";

  if (fromNode.id === node.id) {
    tp = "outgoing";
  } else if (toNode.id === node.id) {
    tp = "incoming";
  }

  return { to: toNode, from: fromNode, type: tp };
};

/**
 * Main logic around dynamic inputs
 *
 * @param {LGraphNode} node - The target node
 * @param {number} index - The slot index of the currently changed connection
 * @param {bool} connected - Was this event connecting or disconnecting
 * @param {string} [connectionPrefix] - The common prefix of the dynamic inputs
 * @param {string|[string]} [connectionType] - The type of the dynamic connection
 * @param {{link?:LLink, ioSlot?:INodeInputSlot | INodeOutputSlot}} [opts] - extra options
 */
const dynamic_connection = (
  node,
  index,
  connected,
  connectionPrefix = "input_",
  connectionType = "*",
  opts = undefined,
) => {
  /* @type {{link?:LLink, ioSlot?:INodeInputSlot | INodeOutputSlot}} [opts] - extra options*/
  const options = opts || {};
  const nameArray = options.nameArray || [];

  // closures used in loops
  const isDynamicInput =
    nameArray.length > 0
      ? (ipt) => {
          return nameArray.includes(ipt.name);
        }
      : (ipt) => {
          return ipt.name.startsWith(connectionPrefix);
        };

  const getName =
    nameArray.length > 0
      ? (i) => {
          return i < nameArray.length
            ? nameArray[i]
            : `${connectionPrefix}${i + 1}`;
        }
      : (i) => {
          return `${connectionPrefix}${i + 1}`;
        };

  // skip non dynamic inputs
  if (node.inputs.length > 0 && !isDynamicInput(node.inputs[index])) {
    return;
  }

  const clean_inputs = () => {
    if (node.inputs.length === 0) return;

    const to_remove = [];
    for (let n = 0; n < node.inputs.length; n++) {
      const element = node.inputs[n];
      if (!isDynamicInput(element)) {
        continue;
      }
      if (!element.link) {
        if (node.widgets) {
          const w = node.widgets.find((w) => w.name === element.name);
          if (w) {
            w.onRemoved?.();
            node.widgets.length = node.widgets.length - 1;
          }
        }
        to_remove.push(n);
      }
    }
    for (let i = 0; i < to_remove.length; i++) {
      const id = to_remove[i];
      node.removeInput(id);
    }
    let w_count = node.widgets?.length || 0;
    let i_count = node.inputs?.length || 0;
    // make inputs sequential again
    for (let i = 0; i < i_count; i++) {
      if (!isDynamicInput(node.inputs[i])) {
        continue;
      }
      const nextIndex = i - node.properties.dynamicInputsIndex;
      const name = getName(nextIndex);

      node.inputs[i].label = name;
      node.inputs[i].name = name;
    }
  };
  if (!connected) {
    if (!options.link) {
      clean_inputs();
    } else {
      if (!options.ioSlot.link) {
        node.connectionTransit = true;
      } else {
        node.connectionTransit = false;
        clean_inputs();
      }
    }
  }

  if (connected) {
    if (options.link) {
      const { from, to, type } = nodesFromLink(node, options.link);
      if (type === "outgoing") return;
    }
    if (node.connectionTransit) {
      node.connectionTransit = false;
    }

    // Remove inputs and their widget if not linked.
    clean_inputs();
    clean_inputs();

    if (node.inputs.length === 0) return;
    // add an extra input
    if (node.inputs[node.inputs.length - 1].link !== null) {
      const nextIndex = node.inputs.length - node.properties.dynamicInputsIndex;
      const name =
        nextIndex < nameArray.length
          ? nameArray[nextIndex]
          : `${connectionPrefix}${nextIndex + 1}`;

      node.addInput(name, connectionType);
    }
  }
};

app.registerExtension({
  name: "CosyVoice.UploadSRT",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData?.name == "LoadSRT") {
      nodeData.input.required.upload = ["SRTPLOAD"];
    }

    if (nodeData?.name == "CosyVoiceDialog") {
      setupDynamicConnections(nodeType, "voice_", "AUDIO", {
        nameArray: Array.from({ length: 26 }, (_, i) =>
          String.fromCharCode(i + 65),
        ),
      });
    }
  },
});
