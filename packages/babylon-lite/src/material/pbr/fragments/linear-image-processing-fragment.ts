import type { PbrExt } from "../pbr-flags.js";
import { wgsl } from "../../../shader/wgsl.js";

const PBR2_LINEAR_IMAGE_PROCESSING = 1 << 14;

/** @internal Keep PBR output linear while a feature owns a final image-processing pass. */
export function makeLinearImageProcessingExt(): PbrExt {
    const slots = { IP_BEGIN: wgsl`if(scene.vImageInfos.w>=0.0){`, IP_END: wgsl`}` };
    return {
        id: "linear-image-processing",
        phase: "fragment",
        detect(material) {
            return { f: 0, f2: (material as { _linearImageProcessing?: boolean })._linearImageProcessing ? PBR2_LINEAR_IMAGE_PROCESSING : 0 };
        },
        frag(context) {
            return context._features2 & PBR2_LINEAR_IMAGE_PROCESSING ? { _id: "linear", _fragmentSlots: slots } : null;
        },
    };
}
