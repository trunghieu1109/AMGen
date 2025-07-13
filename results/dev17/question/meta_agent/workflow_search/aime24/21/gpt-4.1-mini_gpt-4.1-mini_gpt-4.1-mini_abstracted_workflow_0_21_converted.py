async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Formally define the geometric elements of the problem by characterizing the regular dodecagon, "
        "including its vertices, sides, and diagonals. Enumerate these elements explicitly, specifying how vertices are indexed "
        "and how sides and diagonals are represented as chords between vertices."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, defining geometric elements, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, precisely define what it means for a rectangle side to lie on a side or diagonal of the dodecagon. "
        "Clarify assumptions that rectangle vertices must coincide with polygon vertices, that rectangle sides coincide exactly with polygon sides or diagonals (not just lie along the same line), "
        "and that rectangles are convex and non-degenerate."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, clarifying rectangle side definition, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2 = "Given all the above thinking and answers, find the most consistent and correct definition for rectangle sides in this context."
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent answer for rectangle side definition." + final_instr_2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Identify and formalize the geometric conditions that a quadrilateral must satisfy to be a rectangle within the dodecagon context. "
        "Express these conditions in terms of polygon vertices and edges, including perpendicularity of adjacent sides and equality of opposite sides, "
        "without imposing unnecessary restrictions on chord lengths or step differences."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking1.content, thinking2.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, thinking2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, formalizing rectangle conditions, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Analyze the symmetry and angular properties of the regular dodecagon to deduce constraints on which sides and diagonals can form right angles. "
        "Determine all possible orientations of chords that differ by 90 degrees modulo 180 degrees, without restricting to fixed step differences, to identify candidate perpendicular edges."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, analyzing chord orientations, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_4 = "Given all the above thinking and answers, synthesize the constraints on chord orientations for perpendicularity in the dodecagon."
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_thinkings_4, "Sub-task 4: Synthesize chord orientation constraints." + final_instr_4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 5: Enumerate all sides and diagonals of the dodecagon, categorizing them by their length and orientation. "
        "Compute and store orientation data (angles) for each chord to facilitate identification of perpendicular pairs in subsequent steps."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking1], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, enumerating chords and orientations, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction_6 = (
        "Sub-task 6: For each vertex, enumerate all pairs of adjacent edges (chords) whose orientations differ by exactly 90 degrees modulo 180 degrees, regardless of their lengths. "
        "Output all such candidate pairs (vertex i, step differences d1, d2) that can serve as adjacent sides of rectangles, removing any prior assumptions restricting step differences or chord lengths."
    )
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_6 = []
    possible_thinkings_6 = []
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", thinking4.content, thinking5.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking6, answer6 = await cot_agents_6[i]([taskInfo, thinking4, thinking5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, enumerating perpendicular chord pairs, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6)
        possible_thinkings_6.append(thinking6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_6 = "Given all the above thinking and answers, synthesize the complete list of candidate adjacent chord pairs for rectangles."
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + possible_thinkings_6, "Sub-task 6: Synthesize candidate adjacent chord pairs." + final_instr_6, is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", thinking4.content, thinking5.content],
        "agent_collaboration": "SC_CoT",
        "response": {"thinking": thinking6, "answer": answer6}
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    reflect_inst_7 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_7 = (
        "Sub-task 7: For each candidate adjacent edge pair (vertex i, d1, d2) from Subtask 6, compute the opposite vertex by translating by (d1 + d2) modulo 12. "
        "Verify that the chords at this opposite vertex match the same step differences, ensuring opposite sides are equal and parallel. Collect all valid quadruples of vertices that form rectangles, ensuring closure and vertex consistency without degeneracy. "
        + reflect_inst_7
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking6]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", thinking6.content],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, verifying rectangle candidates, thinking: {thinking7.content}; answer: {answer7.content}")
    critic_inst_7 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_7):
        feedback7, correct7 = await critic_agent_7([taskInfo, thinking7], critic_inst_7, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}")
        if correct7.content == "True":
            break
        cot_inputs_7.extend([thinking7, feedback7])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining rectangle verification, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction_8 = (
        "Sub-task 8: Perform an exhaustive enumeration and verification of all quadruples of vertices that could form rectangles. "
        "Explicitly check rectangle conditions—perpendicularity of adjacent sides, equality of opposite sides, and that each side coincides exactly with a polygon side or diagonal—thus ensuring completeness beyond symmetry or modular arithmetic assumptions."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", thinking5.content, thinking6.content, thinking7.content],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking5, thinking6, thinking7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, exhaustive verification of rectangles, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    cot_instruction_9 = (
        "Sub-task 9: Compute the total number of distinct rectangles formed by the identified sets of edges from Subtask 8. "
        "Account for symmetries and avoid double counting by considering equivalence classes under polygon rotations and reflections."
    )
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_instruction_9,
        "context": ["user query", thinking8.content],
        "agent_collaboration": "CoT"
    }
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking8], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, computing total rectangle count, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    reflect_inst_10 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_10 = (
        "Sub-task 10: Verify the computed count by cross-checking with known example rectangles (such as those shown in the diagram) and by validating geometric consistency. "
        "Include a reflective phase where alternative rectangle candidates are proposed, tested, and validated to ensure no valid rectangles are missed. "
        + reflect_inst_10
    )
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_10 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_10 = self.max_round
    cot_inputs_10 = [taskInfo, thinking9]
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": cot_reflect_instruction_10,
        "context": ["user query", thinking9.content],
        "agent_collaboration": "Reflexion"
    }
    thinking10, answer10 = await cot_agent_10(cot_inputs_10, cot_reflect_instruction_10, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_10.id}, verifying final count, thinking: {thinking10.content}; answer: {answer10.content}")
    critic_inst_10 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_10):
        feedback10, correct10 = await critic_agent_10([taskInfo, thinking10], critic_inst_10, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_10.id}, providing feedback, thinking: {feedback10.content}; answer: {correct10.content}")
        if correct10.content == "True":
            break
        cot_inputs_10.extend([thinking10, feedback10])
        thinking10, answer10 = await cot_agent_10(cot_inputs_10, cot_reflect_instruction_10, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_10.id}, refining final verification, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs
