async def forward_190(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = (
        "Sub-task 1: Parse the reaction sequence for 3-(hydroxymethyl)-5-(prop-1-en-2-yl)cyclohexan-1-one; "
        "identify the starting material, each reagent step (NaH/benzyl bromide, tosylhydrazide/HCl, n-BuLi/NH4Cl, Pd/C/H2), "
        "and the type of transformation at each stage."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, parsing sequence, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = (
        "Sub-task 2: Based on sub-task 1 output, deduce the structure of product 1 by showing NaH deprotonates the hydroxymethyl oxygen to form an alkoxide, "
        "and benzyl bromide undergoes SN2 to give a benzyl ether; include a reflexion check confirming O- vs C-alkylation."
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "thinking1", "answer1"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents:
        thinking2_i, answer2_i = await agent([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, reasoning product 1 structure, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings.append(thinking2_i)
        possible_answers.append(answer2_i)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings + possible_answers,
                                                    "Sub-task 2: Synthesize and choose the most consistent structure for product 1.",
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    reflect_inst = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction = (
        "Sub-task 3: Deduce the structure of product 2 from product 1 by forming a p-toluenesulfonyl hydrazone at the ketone, "
        "retaining the benzyl ether;" + reflect_inst
    )
    critic_inst = (
        "Please review the answer above and criticize on where might be wrong. "
        "If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
    )
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_reflect_instruction, "context": ["user query", "thinking1", "answer1", "thinking2", "answer2"], "agent_collaboration": "Reflexion"}
    thinking3, answer3 = await cot_agent3(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, initial product 2 proposal, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent3([taskInfo, thinking3, answer3], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback.content}; correct flag: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs += [thinking3, answer3, feedback]
        thinking3, answer3 = await cot_agent3(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refined product 2, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    debate_instr = (
        "Given solutions to the problem from other agents, consider their opinions as additional advice. "
        "Please think carefully and provide an updated answer."
    )
    debate_instruction_4 = (
        "Sub-task 4: Outline the general mechanism of the Shapiro reaction: base-promoted elimination of N2 and Ts to give a vinyl carbanion intermediate, then protonation to form an alkene." 
        + debate_instr
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": debate_instruction_4, "context": ["user query", "thinking3", "answer3"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1], debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, outlining mechanism, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_4 = "Sub-task 4: Outline the Shapiro mechanism. Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1], final_instr_4, is_sub_task=True)
    agents.append(f"Final Decision agent, mechanism outline, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 5: Apply the Shapiro mechanism to product 2: show loss of N2/Ts, formation of the alkene at C-1, and protonation to yield product 3;" + debate_instr
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": debate_instruction_5, "context": ["user query", "thinking4", "answer4"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1], debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, applying mechanism, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_5 = "Sub-task 5: Apply Shapiro mechanism. Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking5, answer5 = await final_decision_agent_5([taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1], final_instr_5, is_sub_task=True)
    agents.append(f"Final Decision agent, mechanism application, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    debate_instruction_6 = (
        "Sub-task 6: Deduce how Pd/C hydrogenation transforms product 3: cleaves benzyl ether to regenerate free hydroxymethyl, saturates any C=C if present, and match the final structure to the correct choice;" + debate_instr
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking6 = [[] for _ in range(self.max_round)]
    all_answer6 = [[] for _ in range(self.max_round)]
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": debate_instruction_6, "context": ["user query", "thinking5", "answer5"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instruction_6, r, is_sub_task=True)
            else:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1], debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, hydrogenation outcome, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_6 = "Sub-task 6: Hydrogenation outcome. Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking6, answer6 = await final_decision_agent_6([taskInfo, thinking5, answer5] + all_thinking6[-1] + all_answer6[-1], final_instr_6, is_sub_task=True)
    agents.append(f"Final Decision agent, hydrogenation final, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs