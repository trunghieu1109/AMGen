async def forward_4(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the condition p^2 divides n^4 + 1 and characterize the modular arithmetic implications "
        "for prime p and integer n, with input: taskInfo containing the problem statement and detailed analysis."
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

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, determine constraints on prime p and integer n from the congruence n^4 ≡ -1 mod p^2, "
        "and identify properties of the least such prime p. Input includes taskInfo, thinking and answer from stage_0.subtask_1."
    )
    final_decision_instruction_0_2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for constraints on p and n from stage_0.subtask_1 outputs."
    )
    cot_sc_desc_0_2 = {
        "instruction": cot_sc_instruction_0_2,
        "final_decision_instruction": final_decision_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.sc_cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_sc_desc_0_2,
        n_repeat=self.max_sc
    )
    logs.append(log_0_2)

    loop_results_stage_1 = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_sc_instruction_1_1 = (
            f"Iteration {iteration+1} - Sub-task 1: Generate candidate primes p and test existence of n satisfying n^4 ≡ -1 mod p^2, "
            "using inputs: taskInfo and results from stage_0.subtask_2 (all previous thinking and answers)."
        )
        final_decision_instruction_1_1 = (
            f"Iteration {iteration+1} - Sub-task 1: Synthesize and choose the most consistent candidate primes p and n values."
        )
        cot_sc_desc_1_1 = {
            "instruction": cot_sc_instruction_1_1,
            "final_decision_instruction": final_decision_instruction_1_1,
            "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_1"]["answer"] + loop_results_stage_1["stage_1.subtask_1"]["thinking"] + [results_0_2['thinking'], results_0_2['answer']],
            "temperature": 0.5,
            "context_desc": ["user query", "previous answers and thinking of stage_1.subtask_1", "thinking and answer of stage_0.subtask_2"]
        }
        results_1_1, log_1_1 = await self.sc_cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_sc_desc_1_1,
            n_repeat=self.max_sc
        )
        logs.append(log_1_1)
        loop_results_stage_1["stage_1.subtask_1"]["thinking"].append(results_1_1['thinking'])
        loop_results_stage_1["stage_1.subtask_1"]["answer"].append(results_1_1['answer'])

        cot_sc_instruction_1_2 = (
            f"Iteration {iteration+1} - Sub-task 2: Refine candidate primes and corresponding n values by verifying minimality and divisibility conditions, "
            "using inputs: taskInfo and all previous answers and thinking from stage_1.subtask_1."
        )
        final_decision_instruction_1_2 = (
            f"Iteration {iteration+1} - Sub-task 2: Synthesize and choose the most consistent refined candidate primes and n values."
        )
        cot_sc_desc_1_2 = {
            "instruction": cot_sc_instruction_1_2,
            "final_decision_instruction": final_decision_instruction_1_2,
            "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_1"]["answer"] + loop_results_stage_1["stage_1.subtask_1"]["thinking"],
            "temperature": 0.5,
            "context_desc": ["user query", "answers and thinking of stage_1.subtask_1"]
        }
        results_1_2, log_1_2 = await self.sc_cot(
            subtask_id="stage_1.subtask_2",
            cot_agent_desc=cot_sc_desc_1_2,
            n_repeat=self.max_sc
        )
        logs.append(log_1_2)
        loop_results_stage_1["stage_1.subtask_2"]["thinking"].append(results_1_2['thinking'])
        loop_results_stage_1["stage_1.subtask_2"]["answer"].append(results_1_2['answer'])

    aggregate_instruction_2_1 = (
        "Sub-task 1: Evaluate candidate primes and integers from stage_1 to select the least prime p and least positive integer m such that m^4 + 1 is divisible by p^2, "
        "using inputs: taskInfo and all answers and thinking from stage_1.subtask_2."
    )
    aggregate_desc_2_1 = {
        "instruction": aggregate_instruction_2_1,
        "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_2"]["answer"] + loop_results_stage_1["stage_1.subtask_2"]["thinking"],
        "temperature": 0.0,
        "context_desc": ["user query", "answers and thinking of stage_1.subtask_2"]
    }
    results_2_1, log_2_1 = await self.aggregate(
        subtask_id="stage_2.subtask_1",
        aggregate_desc=aggregate_desc_2_1
    )
    logs.append(log_2_1)

    cot_reflect_instruction_3_1 = (
        "Sub-task 1: Verify correctness and consistency of the selected prime p and integer m by checking divisibility and minimality conditions, "
        "using inputs: taskInfo, all answers and thinking from stage_1.subtask_2 and the selected p and m from stage_2.subtask_1."
    )
    critic_instruction_3_1 = (
        "Please review and provide the limitations of provided solutions of selected prime p and integer m, including correctness and minimality checks."
    )
    cot_reflect_desc_3_1 = {
        "instruction": cot_reflect_instruction_3_1,
        "critic_instruction": critic_instruction_3_1,
        "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_2"]["thinking"] + loop_results_stage_1["stage_1.subtask_2"]["answer"] + [results_2_1['thinking'], results_2_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.reflexion(
        subtask_id="stage_3.subtask_1",
        reflect_desc=cot_reflect_desc_3_1,
        n_repeat=self.max_round
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
