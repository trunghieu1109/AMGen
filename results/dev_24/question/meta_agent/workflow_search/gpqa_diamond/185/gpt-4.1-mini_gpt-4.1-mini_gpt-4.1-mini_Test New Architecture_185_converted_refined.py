async def forward_185(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Explicitly extract and model the detailed structure of the starting compound "
        "(1S,4R)-2-vinyl-2-azabicyclo[2.2.1]hept-5-ene, including atom numbering, stereochemistry, and connectivity. "
        "Generate a clear structural representation (e.g., SMILES with stereochemical annotations) to avoid ambiguity. "
        "This step addresses previous failures caused by insufficient stereochemical modeling and ambiguous structural interpretation."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Parse and explicitly map each of the four given product options into detailed structural representations "
        "with atom numbering and stereochemical annotations. Identify hydrogenation patterns, ring fusion types, and positional isomerism by direct interpretation of the IUPAC names into structures. "
        "Cross-verify mappings internally to prevent misreading product nomenclature."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'final_decision_instruction': (
            "Sub-task 2: Synthesize and choose the most consistent and correct structural mappings for the four product options, "
            "given all the above thinking and answers."
        ),
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Perform a rigorous mechanistic analysis of the Cope rearrangement on the starting compound "
        "using the detailed structure from Subtask 1. Track explicitly which bonds break and form by atom number through both chair and boat transition states, "
        "producing tables or diagrams of bond shifts. Consider stereochemical constraints and all possible pathways."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'final_decision_instruction': (
            "Sub-task 3: Synthesize and choose the most consistent mechanistic pathway and bond shift data, "
            "given all the above thinking and answers."
        ),
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Integrate the mechanistic bond-shift data from Subtask 3 with the detailed product structures from Subtask 2. "
        "Cross-validate the stereochemical and connectivity outcomes predicted by the mechanism against each product option’s mapped structure. "
        "Discard product options inconsistent with the mechanistic stereochemical predictions. This integrative step resolves conflicting conclusions and enforces consistency."
    )
    critic_instruction4 = (
        "Please review and provide the limitations of provided solutions for this integration problem, "
        "highlighting any inconsistencies or overlooked stereochemical constraints."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'critic_instruction': critic_instruction4,
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Evaluate and debate the remaining plausible product options from Subtask 4 to select the correct product of the Cope rearrangement. "
        "Synthesize all prior analyses, ensuring the chosen product aligns with both mechanistic and structural evidence. "
        "Critically examine and resolve any residual uncertainties."
    )
    final_decision_instruction5 = "Sub-task 5: Select the correct product of the Cope rearrangement based on the debate."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
