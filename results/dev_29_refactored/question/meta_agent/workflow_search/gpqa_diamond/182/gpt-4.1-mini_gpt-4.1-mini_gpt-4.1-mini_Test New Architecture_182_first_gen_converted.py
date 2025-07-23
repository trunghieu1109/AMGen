async def forward_182(self, taskInfo):
    logs = []
    stage0_results = {"subtask_0": [], "subtask_1": [], "subtask_2": [], "subtask_3": [], "subtask_4": []}

    for iteration in range(3):
        cot_sc_instruction0 = (
            "Sub-task 0: Extract and summarize the given chemical information and reaction conditions relevant to the query."
        )
        cot_sc_desc0 = {
            "instruction": cot_sc_instruction0,
            "final_decision_instruction": "Sub-task 0: Synthesize and choose the most consistent summary of chemical information.",
            "input": [taskInfo],
            "temperature": 0.5,
            "context_desc": ["user query"]
        }
        results0, log0 = await self.sc_cot(
            subtask_id="stage_0.subtask_0_iter_{}".format(iteration),
            cot_agent_desc=cot_sc_desc0,
            n_repeat=self.max_sc
        )
        logs.append(log0)
        stage0_results["subtask_0"].append(results0)

        cot_instruction1 = (
            "Sub-task 1: Analyze the starting compound's structure to determine initial IHD and unsaturation features based on the summary from Sub-task 0."
        )
        cot_agent_desc1 = {
            "instruction": cot_instruction1,
            "input": [taskInfo, results0["thinking"], results0["answer"]],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of subtask 0", "answer of subtask 0"]
        }
        results1, log1 = await self.cot(
            subtask_id="stage_0.subtask_1_iter_{}".format(iteration),
            cot_agent_desc=cot_agent_desc1
        )
        logs.append(log1)
        stage0_results["subtask_1"].append(results1)

        cot_instruction2 = (
            "Sub-task 2: Predict the chemical transformations caused by red phosphorus and excess HI on the functional groups and unsaturations based on Sub-task 1 output."
        )
        cot_agent_desc2 = {
            "instruction": cot_instruction2,
            "input": [taskInfo, results1["thinking"], results1["answer"]],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results2, log2 = await self.cot(
            subtask_id="stage_0.subtask_2_iter_{}".format(iteration),
            cot_agent_desc=cot_agent_desc2
        )
        logs.append(log2)
        stage0_results["subtask_2"].append(results2)

        cot_agent_instruction3 = (
            "Sub-task 3: Generate possible intermediate product structures and their corresponding IHD values based on the predicted transformations from Sub-task 2."
        )
        cot_agent_desc3 = {
            "instruction": cot_agent_instruction3,
            "input": [taskInfo, results2["thinking"], results2["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
        }
        results3, log3 = await self.answer_generate(
            subtask_id="stage_0.subtask_3_iter_{}".format(iteration),
            cot_agent_desc=cot_agent_desc3
        )
        logs.append(log3)
        stage0_results["subtask_3"].append(results3)

        cot_reflect_instruction4 = (
            "Sub-task 4: Refine and simplify the intermediate outputs to produce a clearer, consolidated set of candidate IHD values based on Sub-task 3 outputs."
        )
        critic_instruction4 = (
            "Please review and provide the limitations of provided solutions of candidate IHD values and suggest improvements."
        )
        cot_reflect_desc4 = {
            "instruction": cot_reflect_instruction4,
            "critic_instruction": critic_instruction4,
            "input": [taskInfo, results3["thinking"], results3["answer"]],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of subtask 3", "answer of subtask 3"]
        }
        results4, log4 = await self.reflexion(
            subtask_id="stage_0.subtask_4_iter_{}".format(iteration),
            reflect_desc=cot_reflect_desc4,
            n_repeat=self.max_round
        )
        logs.append(log4)
        stage0_results["subtask_4"].append(results4)

    refined_candidates = [res["answer"] for res in stage0_results["subtask_4"]]

    debate_instruction = (
        "Sub-task 0: Evaluate the refined candidate IHD values against the reaction context and chemical logic to select the best candidate."
    )
    final_decision_instruction = (
        "Sub-task 0: Select the best candidate IHD value based on chemical reasoning and given choices."
    )
    debate_desc = {
        "instruction": debate_instruction,
        "final_decision_instruction": final_decision_instruction,
        "input": [taskInfo] + refined_candidates,
        "context_desc": ["user query"] + ["candidate IHD values from stage_0.subtask_4"] * len(refined_candidates),
        "temperature": 0.5
    }
    results_debate, log_debate = await self.debate(
        subtask_id="stage_1.subtask_0",
        debate_desc=debate_desc,
        n_repeat=self.max_round
    )
    logs.append(log_debate)

    review_instruction = (
        "Sub-task 0: Validate the selected IHD candidate for correctness, consistency, and alignment with chemical principles and the given choices."
    )
    review_desc = {
        "instruction": review_instruction,
        "input": [taskInfo, results_debate["thinking"], results_debate["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_0", "answer of stage_1.subtask_0"]
    }
    results_review, log_review = await self.review(
        subtask_id="stage_2.subtask_0",
        review_desc=review_desc
    )
    logs.append(log_review)

    final_answer = await self.make_final_answer(results_review["thinking"], results_review["answer"])

    return final_answer, logs
