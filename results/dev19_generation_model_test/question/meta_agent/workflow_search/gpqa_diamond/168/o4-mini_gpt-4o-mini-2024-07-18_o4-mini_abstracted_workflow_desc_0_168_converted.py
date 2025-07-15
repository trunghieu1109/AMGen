async def forward_168(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1, Sub-task 1: Extract and summarize decay channels (CoT)
    cot_instruction = (
        "Sub-task 1: Extract and summarize all decay channels, identify particle species A, B, E, V, "
        "and record their rest masses. Note assumptions m_V, m_E, and m_M=0."
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
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 1, Sub-task 2: Clarify definitions (SC-CoT)
    cot_sc_instruction = (
        "Sub-task 2: Clarify definitions: confirm Q refers to the total kinetic energy of both E particles combined, "
        "and note whether E’s are identical or distinguishable."
    )
    N_sc = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                 model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents2:
        t2, a2 = await agent([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {t2.content}; answer: {a2.content}")
        possible_thinkings2.append(t2)
        possible_answers2.append(a2)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    final_inst2 = (
        "Sub-task 2: Synthesize and choose the most consistent and correct solution for definitions."
    )
    thinking2, answer2 = await final_decision_agent2(
        [taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2,
        final_inst2, is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 2, Sub-task 3: Continuity check (SC-CoT)
    cot_sc_instruction3 = (
        "Sub-task 3: Determine if replacing two V’s with one massless M preserves the continuity of the E-spectrum, "
        "referencing phase-space dimensionality."
    )
    cot_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                 model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents3:
        t3, a3 = await agent([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {t3.content}; answer: {a3.content}")
        possible_thinkings3.append(t3)
        possible_answers3.append(a3)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    final_inst3 = (
        "Sub-task 3: Synthesize and choose the most consistent and correct answer for spectral continuity."
    )
    thinking3, answer3 = await final_decision_agent3(
        [taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3,
        final_inst3, is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 3, Sub-task 4: Derive Q_orig (SC-CoT)
    cot_sc_instruction4 = (
        "Sub-task 4: Derive the mass–energy conservation equation for the original decay (2A → 2B + 2E + 2V) "
        "and solve symbolically for Q_orig in terms of M_A, M_B, m_E, and m_V."
    )
    cot_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                 model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction4,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents4:
        t4, a4 = await agent([taskInfo, thinking1, answer1], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {t4.content}; answer: {a4.content}")
        possible_thinkings4.append(t4)
        possible_answers4.append(a4)
    final_decision_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    final_inst4 = (
        "Sub-task 4: Synthesize and choose the most consistent symbolic solution for Q_orig."
    )
    thinking4, answer4 = await final_decision_agent4(
        [taskInfo, thinking1, answer1] + possible_thinkings4 + possible_answers4,
        final_inst4, is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 3, Sub-task 5: Derive Q_new (SC-CoT)
    cot_sc_instruction5 = (
        "Sub-task 5: Derive the mass–energy conservation equation for the variant decay (2A → 2B + 2E + M with m_M=0) "
        "and solve symbolically for Q_new, ensuring m_M=0 is used explicitly."
    )
    cot_agents5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                 model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings5 = []
    possible_answers5 = []
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction5,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents5:
        t5, a5 = await agent([taskInfo, thinking1, answer1], cot_sc_instruction5, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {t5.content}; answer: {a5.content}")
        possible_thinkings5.append(t5)
        possible_answers5.append(a5)
    final_decision_agent5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    final_inst5 = (
        "Sub-task 5: Synthesize and choose the most consistent symbolic solution for Q_new."
    )
    thinking5, answer5 = await final_decision_agent5(
        [taskInfo, thinking1, answer1] + possible_thinkings5 + possible_answers5,
        final_inst5, is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Stage 3, Sub-task 6: Compare endpoints (Debate)
    debate_instr6 = (
        "Given solutions to the problem from other agents, consider their opinions as additional advice. "
        "Please think carefully and provide an updated answer."
    )
    debate_instruction6 = (
        "Sub-task 6: Compare Q_new and Q_orig by computing Δ = Q_new − Q_orig and explicitly show Δ=+2m_V. "
        + debate_instr6
    )
    debate_agents6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                    model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    all_thinking6 = [[] for _ in range(self.max_round)]
    all_answer6 = [[] for _ in range(self.max_round)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction6,
        "context": ["user query", thinking4.content, answer4.content, thinking5.content, answer5.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for agent in debate_agents6:
            if r == 0:
                t6, a6 = await agent(
                    [taskInfo, thinking4, answer4, thinking5, answer5], debate_instruction6, r, is_sub_task=True
                )
            else:
                inputs6 = [taskInfo, thinking4, answer4, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                t6, a6 = await agent(inputs6, debate_instruction6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t6.content}; answer: {a6.content}")
            all_thinking6[r].append(t6)
            all_answer6[r].append(a6)
    final_decision_agent6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    final_instr6 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking6, answer6 = await final_decision_agent6(
        [taskInfo, thinking4, answer4, thinking5, answer5] + all_thinking6[-1] + all_answer6[-1],
        "Sub-task 6: Compare endpoints" + final_instr6, is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    # Stage 4, Sub-task 7: Form final answer (Reflexion)
    reflect_inst7 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction7 = (
        "Sub-task 7: Integrate continuity result and endpoint comparison to form the final answer: "
        "the E-spectrum remains continuous and its endpoint increases by 2m_V. "
        + reflect_inst7
    )
    cot_agent7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                               model=self.node_model, temperature=0.0)
    critic_agent7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                                  model=self.node_model, temperature=0.0)
    cot_inputs7 = [taskInfo, thinking3, answer3, thinking6, answer6]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction7,
        "context": ["user query", thinking3.content, answer3.content, thinking6.content, answer6.content],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent7(cot_inputs7, cot_reflect_instruction7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent7.id}, thinking: {thinking7.content}; answer: {answer7.content}")
    critic_inst7 = (
        "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, "
        "output exactly 'True' in 'correct'"
    )
    for i in range(self.max_round):
        feedback7, correct7 = await critic_agent7(
            [taskInfo, thinking7, answer7], critic_inst7, i, is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent7.id}, feedback: {feedback7.content}; correct: {correct7.content}")
        if correct7.content == "True":
            break
        cot_inputs7.extend([thinking7, answer7, feedback7])
        thinking7, answer7 = await cot_agent7(cot_inputs7, cot_reflect_instruction7, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent7.id}, refining, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs