async def forward_155(self, taskInfo):
    logs = []
    loop_results = {"stage_0.subtask_1": {"thinking": [], "answer": []}, "stage_0.subtask_2": {"thinking": [], "answer": []}}
    for iteration in range(2):
        cot_instruction_1 = (
            "Sub-task 1: Analyze the stereochemical outcome of Reaction 1: epoxidation of (E)-oct-4-ene with mCPBA and subsequent aqueous acid treatment. "
            "Input: taskInfo containing the query describing Reaction 1."
        )
        cot_agent_desc_1 = {
            "instruction": cot_instruction_1,
            "input": [taskInfo],
            "temperature": 0.0,
            "context_desc": ["user query"]
        }
        results1, log1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_1)
        logs.append(log1)
        loop_results["stage_0.subtask_1"]["thinking"].append(results1["thinking"])
        loop_results["stage_0.subtask_1"]["answer"].append(results1["answer"])
        cot_instruction_2 = (
            "Sub-task 2: Analyze the stereochemical outcome of Reaction 2: epoxidation of (Z)-oct-4-ene with mCPBA and subsequent aqueous acid treatment. "
            "Input: taskInfo and all previous thinking and answers from stage_0.subtask_1 iterations."
        )
        cot_agent_desc_2 = {
            "instruction": cot_instruction_2,
            "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results2, log2 = await self.cot(subtask_id="stage_0.subtask_2", cot_agent_desc=cot_agent_desc_2)
        logs.append(log2)
        loop_results["stage_0.subtask_2"]["thinking"].append(results2["thinking"])
        loop_results["stage_0.subtask_2"]["answer"].append(results2["answer"])
    cot_instruction_3 = (
        "Sub-task 1: Combine products from both reactions and determine the number and types of stereoisomers present, considering enantiomers and diastereomers. "
        "Input: taskInfo and all thinking and answers from all iterations of stage_0.subtask_1 and stage_0.subtask_2."
    )
    cot_agent_desc_3 = {
        "instruction": cot_instruction_3,
        "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results3, log3 = await self.cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_agent_desc_3)
    logs.append(log3)
    cot_instruction_4 = (
        "Sub-task 1: Predict the number of peaks observed in standard (achiral) and chiral HPLC chromatograms based on the stereoisomeric mixture and ideal resolution. "
        "Input: taskInfo and thinking and answer from stage_1.subtask_1."
    )
    cot_agent_desc_4 = {
        "instruction": cot_instruction_4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results4, log4 = await self.cot(subtask_id="stage_2.subtask_1", cot_agent_desc=cot_agent_desc_4)
    logs.append(log4)
    final_answer = await self.make_final_answer(results4["thinking"], results4["answer"])
    return final_answer, logs