async def forward_191(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Subtask 1: Extract and organize given quantities
    cot_instruction = (
        "Subtask 1: Extract and organize all given quantities R, r, s, q, L, l, theta and restate the conductor and cavity geometry, "
        "ensuring the condition s + r < R is noted."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                            model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Subtask 2: Debate induced charge symmetry
    debate_instr = (
        "Subtask 2: Debate whether the induced charge on the conductor's outer surface is spherically symmetric "
        "given an off-center cavity and off-center charge; reference that symmetry must be checked rather than assumed."
    )
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent",
                                  model=self.node_model, role=role, temperature=0.5)
                     for role in self.debate_role]
    all_thinking2 = [[] for _ in range(self.max_round)]
    all_answer2 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking2, answer2 = await agent(
                    [taskInfo, thinking1, answer1], debate_instr, r, is_sub_task=True)
            else:
                thinking2, answer2 = await agent(
                    [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1],
                    debate_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_instr2 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                         model=self.node_model, temperature=0.0)
    thinking2_final, answer2_final = await final_decision_agent2(
        [taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1],
        "Subtask 2 final: " + final_instr2,
        is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision_agent2.id}, thinking: {thinking2_final.content}; answer: {answer2_final.content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {thinking2_final.content}; answer - {answer2_final.content}")
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instr,
        "context": ["user query", "output of subtask_1"],
        "agent_collaboration": "Debate",
        "response": {"thinking": thinking2_final, "answer": answer2_final}
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Subtask 3a: Compute induced multipoles via SC-CoT
    cot_sc_instruction_3a = (
        "Subtask 3a: Set up the multipole expansion or method of images for the potential outside the conductor; "
        "compute the induced monopole and dipole moments from the off-center charge +q."
    )
    N = self.max_sc
    cot_sc_agents_3a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                                      model=self.node_model, temperature=0.5)
                        for _ in range(N)]
    possible_thinkings_3a = []
    possible_answers_3a = []
    for i in range(N):
        thinking3a, answer3a = await cot_sc_agents_3a[i](
            [taskInfo, thinking1, answer1, thinking2_final, answer2_final],
            cot_sc_instruction_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3a[i].id}, thinking: {thinking3a.content}; answer: {answer3a.content}")
        possible_thinkings_3a.append(thinking3a)
        possible_answers_3a.append(answer3a)
    final_decision_agent_3a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                           model=self.node_model, temperature=0.0)
    thinking3a_final, answer3a_final = await final_decision_agent_3a(
        [taskInfo, thinking1, answer1, thinking2_final, answer2_final]
        + possible_thinkings_3a + possible_answers_3a,
        "Subtask 3a final: Given the above, select the most consistent induced multipoles.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision_agent_3a.id}, thinking: {thinking3a_final.content}; answer: {answer3a_final.content}")
    sub_tasks.append(f"Subtask 3a output: thinking - {thinking3a_final.content}; answer - {answer3a_final.content}")
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", "output of subtask_1", "output of subtask_2"],
        "agent_collaboration": "SC_CoT",
        "response": {"thinking": thinking3a_final, "answer": answer3a_final}
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])

    # Subtask 3b: Derive explicit field expression via SC-CoT
    cot_sc_instruction_3b = (
        "Subtask 3b: Derive the explicit expression for the electric field magnitude at P in terms of q, l, s, theta "
        "using the multipole result, ensuring the correct angular dependence appears."
    )
    cot_sc_agents_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                                      model=self.node_model, temperature=0.5)
                        for _ in range(N)]
    possible_thinkings_3b = []
    possible_answers_3b = []
    for i in range(N):
        thinking3b, answer3b = await cot_sc_agents_3b[i](
            [taskInfo, thinking1, answer1, thinking2_final, answer2_final,
             thinking3a_final, answer3a_final],
            cot_sc_instruction_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3b[i].id}, thinking: {thinking3b.content}; answer: {answer3b.content}")
        possible_thinkings_3b.append(thinking3b)
        possible_answers_3b.append(answer3b)
    final_decision_agent_3b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                           model=self.node_model, temperature=0.0)
    thinking3b_final, answer3b_final = await final_decision_agent_3b(
        [taskInfo, thinking1, answer1, thinking2_final, answer2_final,
         thinking3a_final, answer3a_final] + possible_thinkings_3b + possible_answers_3b,
        "Subtask 3b final: Synthesize and select the correct field expression.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision_agent_3b.id}, thinking: {thinking3b_final.content}; answer: {answer3b_final.content}")
    sub_tasks.append(f"Subtask 3b output: thinking - {thinking3b_final.content}; answer - {answer3b_final.content}")
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_sc_instruction_3b,
        "context": ["user query", "output of subtask_1", "output of subtask_2", "output of subtask_3a"],
        "agent_collaboration": "SC_CoT",
        "response": {"thinking": thinking3b_final, "answer": answer3b_final}
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])

    # Subtask 4: Reflexion to compare with choices
    reflect_inst = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction = (
        "Subtask 4: Compare the derived field formula from subtask 3b with the four given choices and verify physical limits. "
        + reflect_inst
    )
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                               model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback", "correct"], "Critic Agent",
                                  model=self.node_model, temperature=0.0)
    cot_inputs4 = [taskInfo, thinking1, answer1,
                   thinking2_final, answer2_final,
                   thinking3a_final, answer3a_final,
                   thinking3b_final, answer3b_final]
    thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        feedback4, correct4 = await critic_agent4(
            [taskInfo, thinking4, answer4],
            "Please review the answer above and criticize where it might be wrong. "
            "If you are absolutely sure it is correct, output exactly 'True' in 'correct'.",
            i, is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent4.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content == "True":
            break
        cot_inputs4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, revision thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction,
        "context": ["user query", "outputs of subtasks 1-3"],
        "agent_collaboration": "Reflexion",
        "response": {"thinking": thinking4, "answer": answer4}
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs