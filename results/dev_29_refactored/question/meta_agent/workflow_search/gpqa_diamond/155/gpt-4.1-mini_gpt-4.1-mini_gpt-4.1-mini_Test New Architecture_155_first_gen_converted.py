async def forward_155(self, taskInfo):
    logs = []
    results = {}

    cot_instruction_0_0 = (
        "Sub-task 0: Analyze the stereochemical outcome of Reaction 1: epoxidation of (E)-oct-4-ene with mCPBA and aqueous acid treatment, identifying all possible stereoisomers formed."
    )
    cot_agent_desc_0_0 = {
        "instruction": cot_instruction_0_0,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results["stage_0.subtask_0"], log_0_0 = await self.cot(
        subtask_id="stage_0.subtask_0",
        cot_agent_desc=cot_agent_desc_0_0
    )
    logs.append(log_0_0)

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the stereochemical outcome of Reaction 2: epoxidation of (Z)-oct-4-ene with mCPBA and aqueous acid treatment, identifying all possible stereoisomers formed."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results["stage_0.subtask_1"], log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    debate_instruction_0_2 = (
        "Sub-task 2: Determine the stereochemical relationships (enantiomers, diastereomers, identical species) among the products from Reaction 1 and Reaction 2."
    )
    final_decision_instruction_0_2 = (
        "Sub-task 2: Determine the stereochemical relationships (enantiomers, diastereomers, identical species) among the products from Reaction 1 and Reaction 2."
    )
    debate_desc_0_2 = {
        "instruction": debate_instruction_0_2,
        "final_decision_instruction": final_decision_instruction_0_2,
        "input": [taskInfo, results["stage_0.subtask_0"], results["stage_0.subtask_1"]],
        "context_desc": ["user query", "thinking of stage_0.subtask_0", "thinking of stage_0.subtask_1"],
        "temperature": 0.5
    }
    results["stage_0.subtask_2"], log_0_2 = await self.debate(
        subtask_id="stage_0.subtask_2",
        debate_desc=debate_desc_0_2,
        n_repeat=self.max_round
    )
    logs.append(log_0_2)

    cot_instruction_0_3 = (
        "Sub-task 3: Assess the effect of aqueous acid treatment on the epoxide stereochemistry and product stability to finalize the stereoisomeric structures present."
    )
    cot_agent_desc_0_3 = {
        "instruction": cot_instruction_0_3,
        "input": [taskInfo, results["stage_0.subtask_0"], results["stage_0.subtask_1"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_0", "thinking of stage_0.subtask_1"]
    }
    results["stage_0.subtask_3"], log_0_3 = await self.cot(
        subtask_id="stage_0.subtask_3",
        cot_agent_desc=cot_agent_desc_0_3
    )
    logs.append(log_0_3)

    cot_agent_instruction_1_0 = (
        "Sub-task 0: Combine the stereochemical outcomes and product identities from both reactions into a single comprehensive mixture profile."
    )
    cot_agent_desc_1_0 = {
        "instruction": cot_agent_instruction_1_0,
        "input": [taskInfo, results["stage_0.subtask_2"], results["stage_0.subtask_3"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "thinking of stage_0.subtask_3"]
    }
    results["stage_1.subtask_0"], log_1_0 = await self.answer_generate(
        subtask_id="stage_1.subtask_0",
        cot_agent_desc=cot_agent_desc_1_0
    )
    logs.append(log_1_0)

    cot_instruction_1_1 = (
        "Sub-task 1: Summarize the expected number and types of stereoisomers present in the combined product mixture."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results["stage_1.subtask_0"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_0"]
    }
    results["stage_1.subtask_1"], log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    aggregate_instruction_2_0 = (
        "Sub-task 0: Evaluate which stereoisomers in the mixture are separable by standard (achiral) reverse-phase HPLC based on physicochemical differences (diastereomers vs enantiomers)."
    )
    aggregate_desc_2_0 = {
        "instruction": aggregate_instruction_2_0,
        "input": [taskInfo, results["stage_1.subtask_1"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1"]
    }
    results["stage_2.subtask_0"], log_2_0 = await self.aggregate(
        subtask_id="stage_2.subtask_0",
        aggregate_desc=aggregate_desc_2_0
    )
    logs.append(log_2_0)

    aggregate_instruction_2_1 = (
        "Sub-task 1: Evaluate which stereoisomers in the mixture are separable by chiral HPLC, including both diastereomers and enantiomers."
    )
    aggregate_desc_2_1 = {
        "instruction": aggregate_instruction_2_1,
        "input": [taskInfo, results["stage_1.subtask_1"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1"]
    }
    results["stage_2.subtask_1"], log_2_1 = await self.aggregate(
        subtask_id="stage_2.subtask_1",
        aggregate_desc=aggregate_desc_2_1
    )
    logs.append(log_2_1)

    cot_instruction_2_2 = (
        "Sub-task 2: Validate the total number of chromatographic peaks expected in each chromatogram assuming ideal resolution."
    )
    cot_agent_desc_2_2 = {
        "instruction": cot_instruction_2_2,
        "input": [taskInfo, results["stage_2.subtask_0"], results["stage_2.subtask_1"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_0", "thinking of stage_2.subtask_1"]
    }
    results["stage_2.subtask_2"], log_2_2 = await self.cot(
        subtask_id="stage_2.subtask_2",
        cot_agent_desc=cot_agent_desc_2_2
    )
    logs.append(log_2_2)

    review_instruction_3_0 = (
        "Sub-task 0: Consolidate the chromatographic peak predictions into a clear, concise summary describing the number of peaks observed in standard and chiral HPLC chromatograms."
    )
    review_desc_3_0 = {
        "instruction": review_instruction_3_0,
        "input": [taskInfo, results["stage_2.subtask_2"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_2"]
    }
    results["stage_3.subtask_0"], log_3_0 = await self.review(
        subtask_id="stage_3.subtask_0",
        review_desc=review_desc_3_0
    )
    logs.append(log_3_0)

    formatter_instruction_3_1 = (
        "Sub-task 1: Format the final answer to directly address the query choices, indicating which choice matches the predicted chromatographic outcomes."
    )
    formatter_desc_3_1 = {
        "instruction": formatter_instruction_3_1,
        "input": [taskInfo, results["stage_3.subtask_0"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_3.subtask_0"],
        "format": "short and concise, without explanation"
    }
    results["stage_3.subtask_1"], log_3_1 = await self.specific_format(
        subtask_id="stage_3.subtask_1",
        formatter_desc=formatter_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results["stage_3.subtask_1"].get('thinking', ''), results["stage_3.subtask_1"].get('answer', ''))

    return final_answer, logs
