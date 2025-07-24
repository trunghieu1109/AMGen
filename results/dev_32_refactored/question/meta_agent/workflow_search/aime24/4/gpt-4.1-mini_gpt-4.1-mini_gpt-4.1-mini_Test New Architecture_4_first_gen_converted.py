async def forward_4(self, taskInfo):
    logs = []

    # stage_0.subtask_1
    cot_instruction_0_1 = (
        "Stage 0, Sub-task 1: Analyze the condition for p^2 dividing n^4+1, including modular arithmetic properties and constraints on p and n. "
        "Input content: taskInfo"
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

    loop_results_stage_1 = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(3):
        cot_sc_instruction_1_1 = (
            "Stage 1, Sub-task 1: Generate candidate primes p and test existence of n such that p^2 divides n^4+1, "
            "using analysis from stage_0.subtask_1 and former iterations of stage_1.subtask_2. "
            "Input content: results from stage_0.subtask_1 and all previous iterations of stage_1.subtask_2."
        )
        final_decision_instruction_1_1 = (
            "Stage 1, Sub-task 1, Final Decision: Synthesize and choose the most consistent candidate primes and n values."
        )
        cot_sc_desc_1_1 = {
            "instruction": cot_sc_instruction_1_1,
            "final_decision_instruction": final_decision_instruction_1_1,
            "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]] + loop_results_stage_1["stage_1.subtask_2"]["thinking"] + loop_results_stage_1["stage_1.subtask_2"]["answer"],
            "temperature": 0.6,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of previous stage_1.subtask_2", "answer of previous stage_1.subtask_2"]
        }
        results_1_1, log_1_1 = await self.sc_cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_sc_desc_1_1,
            n_repeat=self.max_sc
        )
        logs.append(log_1_1)

        loop_results_stage_1["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results_stage_1["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        cot_sc_instruction_1_2 = (
            "Stage 1, Sub-task 2: Refine candidate primes and corresponding n values by simplifying and verifying divisibility conditions. "
            "Input content: results from stage_1.subtask_1."
        )
        final_decision_instruction_1_2 = (
            "Stage 1, Sub-task 2, Final Decision: Synthesize and choose the most consistent refined candidates."
        )
        cot_sc_desc_1_2 = {
            "instruction": cot_sc_instruction_1_2,
            "final_decision_instruction": final_decision_instruction_1_2,
            "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
            "temperature": 0.7,
            "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
        }
        results_1_2, log_1_2 = await self.sc_cot(
            subtask_id="stage_1.subtask_2",
            cot_agent_desc=cot_sc_desc_1_2,
            n_repeat=self.max_sc
        )
        logs.append(log_1_2)

        loop_results_stage_1["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results_stage_1["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])

    debate_instruction_2_1 = (
        "Stage 2, Sub-task 1: From refined candidates, select the least prime p and find the least positive integer m such that m^4+1 is divisible by p^2. "
        "Input content: results from stage_0.subtask_1 and all iterations of stage_1.subtask_2."
    )
    final_decision_instruction_2_1 = (
        "Stage 2, Sub-task 1, Final Decision: Select the least prime p and least positive integer m satisfying the problem conditions."
    )
    debate_desc_2_1 = {
        "instruction": debate_instruction_2_1,
        "final_decision_instruction": final_decision_instruction_2_1,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]] + loop_results_stage_1["stage_1.subtask_2"]["thinking"] + loop_results_stage_1["stage_1.subtask_2"]["answer"],
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_2_1,
        n_repeat=self.max_round
    )
    logs.append(log_2_1)

    cot_reflect_instruction_3_1 = (
        "Stage 3, Sub-task 1: Verify correctness and consistency of p and m by checking divisibility and problem criteria. "
        "Input content: results from stage_1.subtask_2 and stage_2.subtask_1."
    )
    critic_instruction_3_1 = (
        "Stage 3, Sub-task 1, Criticism: Review and provide limitations or errors in the solutions for p and m."
    )
    cot_reflect_desc_3_1 = {
        "instruction": cot_reflect_instruction_3_1,
        "critic_instruction": critic_instruction_3_1,
        "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_2"]["thinking"] + loop_results_stage_1["stage_1.subtask_2"]["answer"] + [results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.reflexion(
        subtask_id="stage_3.subtask_1",
        reflect_desc=cot_reflect_desc_3_1,
        n_repeat=self.max_round
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs
