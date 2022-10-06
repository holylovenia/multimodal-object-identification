# ambiguous-mm-dialogue
DSTC 11 | The Third SIMMC Challenge Track 1: Ambiguous Candidate Identification for Immersive Multimodal Conversations

## Important URLs
- [DSTC 11 track proposal / SIMMC 2.1](https://drive.google.com/file/d/1_Tdl7CXm71gqlWutbOe0e8O1hhiycsQf/view)
- [SIMMC 2.0 paper](https://arxiv.org/abs/2104.08667)
- [SIMMC GitHub](https://github.com/facebookresearch/simmc2): including [v2.1 dataset](https://github.com/facebookresearch/simmc2/tree/main/data) used for DSTC 11, 

## DSTC 11 Timeline

| Sub-Task #1 | [Ambiguous Candidate Identification (New)](model/ambiguous_candidates) |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal | Given ambiguous object mentions, to resolve referent objects to thier canonical ID(s). |
| Input | Current user utterance, Dialog context, Multimodal context |
| Output |  Canonical object IDs |
| Metrics | Object Identification F1 / Precision / Recall |

Please check the [task input](https://github.com/facebookresearch/simmc2/blob/main/TASK_INPUTS.md) file for a full description of inputs for each subtask.

## Baseline

| Subtask | Name | Baseline Results | 
| :--: | :--: | :--: |
| #1 | Ambiguous Candidate Identification | [Link](https://github.com/facebookresearch/simmc2/blob/main/model/ambiguous_candidates#performance-on-simmc-21) |
