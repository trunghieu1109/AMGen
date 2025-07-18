async def forward_167(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction_stage0_sub1 = (
        "Sub-task 1: Extract and summarize the four specific issues related to genomics data analysis and the given answer choices, clarifying their definitions and contexts."
    )
    cot_agent_desc_stage0_sub1 = {
        "instruction": cot_instruction_stage0_sub1,
        "input": [taskInfo],
        "temperature": 0.5,
        "context_desc": ["user query"]
    }
    results_stage0_sub1, log_stage0_sub1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage0_sub1)

    cot_instruction_stage1_sub1 = (
        "Sub-task 1: Analyze the relationships and interconnections between the four issues, focusing on how each can cause difficult-to-spot errors in genomics workflows."
    )
    cot_agent_desc_stage1_sub1 = {
        "instruction": cot_instruction_stage1_sub1,
        "input": [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_stage1_sub1, log_stage1_sub1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_stage1_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1_sub1)

    cot_reflect_instruction_stage1_sub2 = (
        "Sub-task 2: Integrate domain knowledge about the prevalence and impact of each issue in typical genomics data analysis pipelines to assess their relative commonality and error subtlety."
    )
    critic_instruction_stage1_sub2 = (
        "Please review and provide the limitations of provided solutions regarding the prevalence and impact of the four issues in genomics data analysis."
    )
    cot_reflect_desc_stage1_sub2 = {
        "instruction": cot_reflect_instruction_stage1_sub2,
        "critic_instruction": critic_instruction_stage1_sub2,
        "input": [
            taskInfo,
            results_stage0_sub1['thinking'], results_stage0_sub1['answer'],
            results_stage1_sub1['thinking'], results_stage1_sub1['answer']
        ],
        "temperature": 0.0,
        "context_desc": [
            "user query",
            "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1",
            "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"
        ]
    }
    results_stage1_sub2, log_stage1_sub2 = await self.reflexion(
        subtask_id="stage_1.subtask_2",
        reflect_desc=cot_reflect_desc_stage1_sub2,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub2)

    debate_instruction_stage2_sub1 = (
        "Sub-task 1: Evaluate and select which combination of issues (from the given answer choices) best represents the most common sources of difficult-to-spot erroneous results, based on the integrated analysis from previous subtasks."
    )
    final_decision_instruction_stage2_sub1 = (
        "Sub-task 1: Select the best answer choice representing the most common sources of difficult-to-spot erroneous results in genomics data analysis."
    )
    debate_desc_stage2_sub1 = {
        "instruction": debate_instruction_stage2_sub1,
        "final_decision_instruction": final_decision_instruction_stage2_sub1,
        "input": [
            taskInfo,
            results_stage1_sub1['thinking'], results_stage1_sub1['answer'],
            results_stage1_sub2['thinking'], results_stage1_sub2['answer']
        ],
        "context_desc": [
            "user query",
            "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1",
            "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"
        ],
        "temperature": 0.5
    }
    results_stage2_sub1, log_stage2_sub1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_stage2_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub1)

    final_answer = await self.make_final_answer(
        results_stage2_sub1['thinking'],
        results_stage2_sub1['answer']
    )

    return final_answer, logs
