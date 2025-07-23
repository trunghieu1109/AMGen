async def forward_159(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Summarize and interpret the given physical setup and geometry, including the polygonal aperture with equal apothems and the incident monochromatic light, and clarify the limit as N approaches infinity."
    )
    cot_sc_desc1 = {
        "instruction": cot_sc_instruction1,
        "final_decision_instruction": "Sub-task 1: Synthesize and choose the most consistent interpretation of the physical setup and limit N->infinity.",
        "input": [taskInfo],
        "temperature": 0.5,
        "context_desc": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Analyze the diffraction pattern characteristics for a regular N-sided polygonal aperture and understand how the pattern evolves as N increases, approaching the circular aperture diffraction pattern, based on Sub-task 1 output."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": "Sub-task 2: Synthesize and choose the most consistent understanding of diffraction pattern evolution as N->infinity.",
        "input": [taskInfo, results1['thinking'], results1['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Derive or recall the angular positions of the first two minima in the far-field diffraction pattern for a circular aperture of radius a, using the small-angle approximation, based on Sub-task 2 output."
    )
    final_decision_instruction3 = "Sub-task 3: Decide the angular positions of the first two minima for the circular aperture diffraction pattern."
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2['thinking'], results2['answer']],
        "context_desc": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Calculate the angular distance between the first two minima for the circular aperture diffraction pattern and express it in terms of wavelength lambda and apothem length a, based on Sub-task 3 output."
    )
    critic_instruction4 = (
        "Please review and provide the limitations of the provided solutions for calculating the angular distance between the first two minima in the circular aperture diffraction pattern."
    )
    cot_reflect_desc4 = {
        "instruction": cot_reflect_instruction4,
        "critic_instruction": critic_instruction4,
        "input": [taskInfo, results1['thinking'], results1['answer'], results3['thinking'], results3['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_instruction5 = (
        "Sub-task 5: Compare the calculated angular distance with the given multiple-choice options and select the correct answer, based on Sub-task 4 output."
    )
    cot_agent_desc5 = {
        "instruction": cot_instruction5,
        "input": [taskInfo, results4['thinking'], results4['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_agent_desc5
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
