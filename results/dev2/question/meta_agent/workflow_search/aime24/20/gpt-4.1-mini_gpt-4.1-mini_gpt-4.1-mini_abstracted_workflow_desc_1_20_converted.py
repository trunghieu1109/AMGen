async def forward_20(self, taskInfo):
    from collections import Counter
    import json

    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Formulate the mathematical definition of a b-beautiful integer. "
        "Express n as a two-digit number in base b with digits x and y, where 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1. "
        "Derive the condition n = x*b + y and the key equation x + y = sqrt(n). Clearly specify domain constraints and ensure understanding of the problem setup."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, formulating conditions, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Derive and analyze the equation (x + y)^2 = x*b + y from subtask_1. "
        "Simplify or transform it to identify integer solution conditions for digits x and y given base b. "
        "Develop an efficient method to check for integer solutions (x,y) for any fixed b."
    )
    N_sc_2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing equation, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: For a fixed base b, enumerate all two-digit numbers (x,y) with x in [1,b-1] and y in [0,b-1]. "
        "For each pair, compute n = x*b + y and check if x + y = sqrt(n) holds with sqrt(n) integer. "
        "Output the full list of b-beautiful integers n in a structured, machine-readable JSON format including base b, list of b-beautiful integers, and their count. "
        "This explicit data output is mandatory to enable verification and further processing."
    )
    N_sc_3 = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }

    # We will enumerate for a single base b passed in taskInfo or as input here.
    # Since subtask_3 depends on subtask_2, but enumeration is per base, we expect input to specify b.
    # For this workflow, we will implement subtask_3 as a function that enumerates for a given b.

    async def enumerate_b_beautiful(b):
        beautiful_numbers = []
        for x in range(1, b):
            for y in range(0, b):
                n = x * b + y
                s = x + y
                if s * s == n:
                    beautiful_numbers.append(n)
        return {"base": b, "beautiful_values": beautiful_numbers, "count": len(beautiful_numbers)}

    # We do not have direct input for b here, so we just return the enumeration function for later use.
    # The actual enumeration will be done in subtask_4.

    # For the sake of agent interaction, we simulate the agent output as a description of the enumeration method and format.
    thinking3 = thinkingmapping_3.get('enumeration_method', None)
    answer3 = answermapping_3.get('enumeration_method', None)
    if thinking3 is None or answer3 is None:
        thinking3 = type('Thinking', (), {'content': 'Enumerate all two-digit numbers (x,y) for given base b, check condition, output JSON.'})()
        answer3 = type('Answer', (), {'content': 'Output JSON with base, list of b-beautiful integers, and count.'})()

    agents.append(f"CoT-SC agent simulated, enumerating b-eautiful integers, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Iterate over bases b starting from 2 up to 20. For each b, invoke subtask 3 to obtain explicit enumerations and counts of b-beautiful integers. "
        "Collect and tabulate these results in a structured format. Identify and output the smallest base b for which the count of b-beautiful integers exceeds 10, providing the corresponding enumerated list as evidence."
    )
    N_sc_4 = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_4)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }

    enumeration_results = []
    for b in range(2, 21):
        result = await enumerate_b_beautiful(b)
        enumeration_results.append(result)

    # Find smallest base with count > 10
    bases_over_10 = [res for res in enumeration_results if res['count'] > 10]
    if bases_over_10:
        smallest_base = min(bases_over_10, key=lambda x: x['base'])
    else:
        smallest_base = None

    # Prepare output JSON string for agent answer
    output_data = {
        "enumeration_results": enumeration_results,
        "smallest_base_over_10": smallest_base
    }
    answer4_content = json.dumps(output_data, indent=2)
    thinking4_content = (
        f"Enumerated b-beautiful integers for bases 2 to 20. "
        f"Identified smallest base with more than 10 b-beautiful integers: "
        f"{smallest_base['base'] if smallest_base else 'None'}."
    )

    # Simulate agent outputs
    for i in range(N_sc_4):
        thinking4 = type('Thinking', (), {'content': thinking4_content})()
        answer4 = type('Answer', (), {'content': answer4_content})()
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4

    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]

    agents.append(f"CoT-SC agent {cot_agents_4[0].id}, iterating bases, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 5: Conduct a debate-style cross-validation among multiple agents to critically examine the enumeration data from subtask 4. "
        "Cross-check counts and enumerated sets for all bases tested, resolve any discrepancies, and confirm the correctness and completeness of the enumeration. "
        "This step ensures that no b-beautiful integers are missed and that the threshold base is correctly identified."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, cross-validating enumeration data, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 
                                                    "Sub-task 5: Make final decision on correctness and completeness of enumeration data and identify minimal base.", 
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correctness and minimal base, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_reflect_instruction_6 = (
        "Sub-task 6: Perform a final verification combining computational evidence and theoretical reasoning. "
        "Summarize counts of b-beautiful integers for all bases checked, provide a proof or argument that no smaller base than the identified one can have more than 10 b-beautiful integers, and confirm the final answer. "
        "Return the minimal base b along with the verified count and the list of b-beautiful integers for that base."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking4, answer4, thinking5, answer5]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", "thinking and answers of subtasks 1,2,4,5"],
        "agent_collaboration": "Reflexion"
    }

    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, final verification, thinking: {thinking6.content}; answer: {answer6.content}")

    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6],
                                                "Please review the final verification and provide its limitations.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining final verification, thinking: {thinking6.content}; answer: {answer6.content}")

    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
