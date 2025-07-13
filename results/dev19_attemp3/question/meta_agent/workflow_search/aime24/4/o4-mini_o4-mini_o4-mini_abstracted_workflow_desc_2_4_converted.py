async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: Chain-of-Thought
    cot_instruction = "Sub-task 1: Formalize n^4+1 divisible by p^2 as congruence n^4 ≡ -1 mod p^2 and deduce reduction mod p."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, formalizing congruence, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Self-Consistency Chain-of-Thought
    cot_sc_instruction = "Sub-task 2: Based on outputs of Sub-task 1, analyze X^4 ≡ -1 mod p to find possible primes p."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, analyzing primes, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings.append(thinking2_i)
        possible_answers.append(answer2_i)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr2 = "Sub-task 2: Synthesize and choose the most consistent answer for possible primes p. Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings + possible_answers, final_instr2, is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent_2.id}, selecting primes, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Debate
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_3 = "Sub-task 3: Determine which candidate primes allow lifting a solution from mod p to mod p^2 using Hensel lemma-type reasoning." + debate_instr
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(self.max_round)]
    all_answer3 = [[] for _ in range(self.max_round)]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": debate_instruction_3, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2], debate_instruction_3, r, is_sub_task=True)
            else:
                inputs3 = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3_i, answer3_i = await agent(inputs3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
            all_thinking3[r].append(thinking3_i)
            all_answer3[r].append(answer3_i)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], final_instr3, is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent_3.id}, deciding liftability, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Debate
    debate_instruction_4 = "Sub-task 4: Select the smallest prime p among those identified in stage_0 for which n^4 ≡ -1 mod p^2 has a solution." + debate_instr
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": debate_instruction_4, "context": ["user query", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4_i, answer4_i = await agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                inputs4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4_i, answer4_i = await agent(inputs4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
            all_thinking4[r].append(thinking4_i)
            all_answer4[r].append(answer4_i)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr4 = "Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1], final_instr4, is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent_4.id}, selecting smallest p, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Self-Consistency Chain-of-Thought
    cot_sc_instruction5 = "Sub-task 5: Based on prime p, compute the least positive integer m such that m^4 ≡ -1 mod p^2."
    N5 = self.max_sc
    cot_agents5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible_thinkings5 = []
    possible_answers5 = []
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_sc_instruction5, "context": ["user query", "thinking of subtask 4", "answer of subtask 4"], "agent_collaboration": "SC_CoT"}
    for i in range(N5):
        thinking5_i, answer5_i = await cot_agents5[i]([taskInfo, thinking4, answer4], cot_sc_instruction5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents5[i].id}, finding m, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
        possible_thinkings5.append(thinking5_i)
        possible_answers5.append(answer5_i)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr5 = "Sub-task 5: Synthesize and choose the most consistent answer for m. Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking5, answer5 = await final_decision_agent_5([taskInfo, thinking4, answer4] + possible_thinkings5 + possible_answers5, final_instr5, is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent_5.id}, selecting m, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs