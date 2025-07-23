async def forward_10(self, taskInfo):
    logs = []
    loop_results = {"stage_0.subtask_1": {"thinking": [], "answer": []},
                    "stage_0.subtask_2": {"thinking": [], "answer": []},
                    "stage_1.subtask_1": {"thinking": [], "answer": []},
                    "stage_2.subtask_1": {"thinking": [], "answer": []}}

    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Extract and formalize the geometric properties of rectangles ABCD and EFGH, "
            "including side lengths and right angles. Input: [taskInfo]"
        )
        cot_agent_desc_0_1 = {
            "instruction": cot_instruction_0_1,
            "input": [taskInfo],
            "temperature": 0.0,
            "context_desc": ["user query"]
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id="stage_0.subtask_1",
            cot_agent_desc=cot_agent_desc_0_1
        )
        logs.append(log_0_1)
        loop_results["stage_0.subtask_1"]["thinking"].append(results_0_1["thinking"])
        loop_results["stage_0.subtask_1"]["answer"].append(results_0_1["answer"])

        cot_instruction_0_2 = (
            "Sub-task 2: Analyze the collinearity of points D, E, C, F and the concyclicity of points A, D, H, G "
            "to establish geometric constraints. Input: [taskInfo, thinking and answer from stage_0.subtask_1]"
        )
        cot_agent_desc_0_2 = {
            "instruction": cot_instruction_0_2,
            "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id="stage_0.subtask_2",
            cot_agent_desc=cot_agent_desc_0_2
        )
        logs.append(log_0_2)
        loop_results["stage_0.subtask_2"]["thinking"].append(results_0_2["thinking"])
        loop_results["stage_0.subtask_2"]["answer"].append(results_0_2["answer"])

        cot_sc_instruction_1_1 = (
            "Sub-task 1: Apply the constraints from stage_0 to set up equations and solve for the length CE. "
            "Input: [taskInfo, thinking and answer from stage_0.subtask_2, and previous iterations of stage_1.subtask_1]"
        )
        final_decision_instruction_1_1 = (
            "Sub-task 1: Synthesize and choose the most consistent answer for the length CE."
        )
        cot_sc_desc_1_1 = {
            "instruction": cot_sc_instruction_1_1,
            "final_decision_instruction": final_decision_instruction_1_1,
            "input": [taskInfo] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"] + loop_results["stage_1.subtask_1"]["thinking"] + loop_results["stage_1.subtask_1"]["answer"],
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of previous stage_1.subtask_1", "answer of previous stage_1.subtask_1"]
        }
        results_1_1, log_1_1 = await self.sc_cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_sc_desc_1_1,
            n_repeat=self.max_sc
        )
        logs.append(log_1_1)
        loop_results["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        cot_instruction_2_1 = (
            "Sub-task 1: Assess the computed length CE for consistency with all given conditions and select the valid solution. "
            "Input: [taskInfo, thinking and answer from stage_0.subtask_2, thinking and answer from stage_1.subtask_1]"
        )
        cot_agent_desc_2_1 = {
            "instruction": cot_instruction_2_1,
            "input": [taskInfo] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"] + loop_results["stage_1.subtask_1"]["thinking"] + loop_results["stage_1.subtask_1"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
        }
        results_2_1, log_2_1 = await self.cot(
            subtask_id="stage_2.subtask_1",
            cot_agent_desc=cot_agent_desc_2_1
        )
        logs.append(log_2_1)
        loop_results["stage_2.subtask_1"]["thinking"].append(results_2_1["thinking"])
        loop_results["stage_2.subtask_1"]["answer"].append(results_2_1["answer"])

    final_thinking = loop_results["stage_2.subtask_1"]["thinking"][-1]
    final_answer = loop_results["stage_2.subtask_1"]["answer"][-1]

    final_answer = await self.make_final_answer(final_thinking, final_answer)
    return final_answer, logs
