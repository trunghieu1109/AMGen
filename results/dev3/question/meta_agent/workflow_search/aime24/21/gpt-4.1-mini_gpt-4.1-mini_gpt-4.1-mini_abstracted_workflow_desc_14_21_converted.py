async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Identify and enumerate all relevant line segments in the regular dodecagon, including its 12 sides and all diagonals connecting pairs of vertices. "
        "Explicitly clarify and confirm the interpretation of 'lying on a side or diagonal' to include that rectangle sides may coincide with full polygon edges, full diagonals, or subsegments thereof. "
        "Establish assumptions that rectangle vertices can be polygon vertices or intersection points of diagonals inside the polygon. "
        "Define counting criteria for distinct rectangles (e.g., by vertex sets or geometric uniqueness). Avoid restrictive assumptions that limit vertices to polygon vertices only."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_1: {subtask_desc1}")
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying segments and clarifying assumptions, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Enumerate all intersection points formed by pairs of diagonals inside the polygon. "
        "For each pair of diagonals, compute their intersection point if it lies strictly inside the polygon. "
        "Collect these intersection points and pass them forward_21 as potential rectangle vertices. "
        "Ensure that intersection points are computed precisely and duplicates are removed."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask_1", "answer of subtask_1"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_2: {subtask_desc2}")
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, enumerating diagonal intersections, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Analyze the geometric properties of all identified points (polygon vertices and diagonal intersection points) and segments formed between them that lie on polygon sides or diagonals (including subsegments). "
        "Determine directions (angles), lengths, and classify these segments by direction to identify candidates for parallel sides of rectangles. "
        "Prepare data structures to efficiently check perpendicularity and parallelism conditions for rectangle formation. "
        "Emphasize that segments may be subsegments between intersection points and vertices, not only full polygon edges or diagonals."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask_2", "answer of subtask_2"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_3: {subtask_desc3}")
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing segments and directions, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Derive formal geometric criteria for rectangle formation applicable to quadrilaterals formed by any combination of polygon vertices and intersection points. "
        "Specifically, identify sets of four points that form closed quadrilaterals with four right angles, opposite sides parallel and equal in length, and sides lying on polygon sides or diagonals (including subsegments). "
        "Clarify and formalize how to verify these conditions using the directions and lengths computed previously. "
        "Ensure criteria allow for rectangles whose sides are subsegments of polygon edges or diagonals."
    )
    N_sc = self.max_sc
    cot_sc_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask_3", "answer of subtask_3"],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before subtask_4: {subtask_desc4}")
    for i in range(N_sc):
        thinking4, answer4 = await cot_sc_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_4[i].id}, deriving rectangle criteria, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    best_answer_4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[best_answer_4]
    answer4 = answermapping_4[best_answer_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_5 = (
        "Sub-task 5: Enumerate all rectangles satisfying the criteria from Sub-task 4. "
        "Implement a deterministic, step-by-step enumeration method that explicitly lists candidate quadruples of points and verifies rectangle conditions. "
        "Avoid majority voting or heuristic shortcuts. For each chord-step class (e.g., a in {1,2,4,5,6}), compute the number of rectangles using arithmetic formulas involving gcd and verify counts by explicit enumeration or pseudocode. "
        "Include an internal self-verification step by testing small cases (e.g., a=1) to ensure no duplicates or omissions. "
        "Record all valid rectangles and sum counts to obtain the total number of rectangles formed inside the dodecagon."
    )
    cot_sc_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_5 = []
    thinkingmapping_5 = {}
    answermapping_5 = {}
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", "thinking of subtask_4", "answer of subtask_4"],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before subtask_5: {subtask_desc5}")
    for i in range(N_sc):
        thinking5, answer5 = await cot_sc_agents_5[i]([taskInfo, thinking4, answer4], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_5[i].id}, enumerating rectangles, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5.content)
        thinkingmapping_5[answer5.content] = thinking5
        answermapping_5[answer5.content] = answer5
    best_answer_5 = Counter(possible_answers_5).most_common(1)[0][0]
    thinking5 = thinkingmapping_5[best_answer_5]
    answer5 = answermapping_5[best_answer_5]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    reflect_instruction_6 = (
        "Sub-task 6: Cross-check the enumeration results with alternative methods such as computational geometry algorithms, known mathematical results, or combinatorial arguments that consider intersection points. "
        "Implement internal sanity checks comparing computed totals with quick script enumerations or smaller polygon cases. Flag any discrepancies for human review. "
        "Refine the final count if needed. Provide a final verified count of rectangles formed inside the regular dodecagon with sides on polygon edges or diagonals (including subsegments)."
    )
    cot_reflect_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": reflect_instruction_6,
        "context": ["user query", "thinking of subtask_5", "answer of subtask_5"],
        "agent_collaboration": "Reflexion"
    }
    print(f"Logging before subtask_6: {subtask_desc6}")
    thinking6, answer6 = await cot_reflect_agent_6(cot_inputs_6, reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_reflect_agent_6.id}, verifying enumeration, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback6, correct6 = await critic_agent_6([taskInfo, thinking6, answer6],
                                                 "Please review and provide limitations or confirm correctness. Output exactly 'True' if correct.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, feedback round {i}, thinking: {feedback6.content}; answer: {correct6.content}")
        if correct6.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = await cot_reflect_agent_6(cot_inputs_6, reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_reflect_agent_6.id}, refining verification, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    debate_instruction_7 = (
        "Sub-task 7: Synthesize the initial enumeration and verification results to produce a final answer. "
        "Address any verification feedback or refinements. Present the final total number of rectangles along with a summary of the verification process and confidence in correctness. "
        "This subtask ensures that the final output is both accurate and well-justified."
    )
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": debate_instruction_7,
        "context": ["user query", "thinking of subtask_6", "answer of subtask_6"],
        "agent_collaboration": "Debate"
    }
    print(f"Logging before subtask_7: {subtask_desc7}")
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6, answer6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, synthesizing final answer, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Final synthesis and answer." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, producing final rectangle count, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
