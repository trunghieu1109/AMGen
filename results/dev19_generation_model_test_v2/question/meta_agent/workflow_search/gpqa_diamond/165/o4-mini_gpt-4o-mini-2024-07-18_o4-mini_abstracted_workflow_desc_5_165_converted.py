async def forward_165(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Subtask 1: SC_CoT to extract and summarize model features
    N1 = self.max_sc
    sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    instruction1 = "Subtask 1: Extract and summarize the model’s defining features including field content, gauge charges, kinetic and Yukawa interactions, VEVs x and v, and the target pseudo-Goldstone H2 from the given query."
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": instruction1, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    possible_thinkings1 = []
    possible_answers1 = []
    for agent in sc_agents_1:
        thinking, answer = await agent([taskInfo], instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings1.append(thinking)
        possible_answers1.append(answer)
    final_decider1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decider1([taskInfo] + possible_thinkings1 + possible_answers1, "Subtask 1 decision: Synthesize the most consistent summary of model features.", is_sub_task=True)
    sub_tasks.append(f"Subtask 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Subtask 2: SC_CoT to identify one-loop contributions
    N2 = self.max_sc
    sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    instruction2 = "Subtask 2: Identify and classify all one-loop contributions to M_h2^2, distinguishing bosonic loops (scalars, W, Z) from fermionic loops (top quark, singlet neutrinos)."
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": instruction2, "context": ["user query", thinking1.content, answer1.content], "agent_collaboration": "SC_CoT"}
    possible_thinkings2 = []
    possible_answers2 = []
    for agent in sc_agents_2:
        thinking, answer = await agent([taskInfo, thinking1, answer1], instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings2.append(thinking)
        possible_answers2.append(answer)
    final_decider2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decider2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Subtask 2 decision: Synthesize the most consistent classification of loop contributions.", is_sub_task=True)
    sub_tasks.append(f"Subtask 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Subtask 3: Debate to construct Coleman-Weinberg expression
    debate_instr3 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    instruction3 = "Subtask 3: Construct the general Coleman-Weinberg one-loop mass correction expression for H2 and map each mass term and sign to the structures in choices 1-4. " + debate_instr3
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": instruction3, "context": ["user query", thinking2.content, answer2.content], "agent_collaboration": "Debate"}
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(self.max_round)]
    all_answer3 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for agent in debate_agents_3:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking2, answer2], instruction3, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking, answer = await agent(inputs, instruction3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking3[r].append(thinking)
            all_answer3[r].append(answer)
    final3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], "Subtask 3 decision: Given all the above thinking and answers, reason over them carefully and provide a final expression for the one-loop correction.", is_sub_task=True)
    sub_tasks.append(f"Subtask 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Subtask 4: Debate to select correct candidate formula
    debate_instr4 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    instruction4 = "Subtask 4: Evaluate the four candidate formulas against the derived expression—checking the presence of the top-quark term, the neutrino sum, sign conventions, and the overall factor—and select the correct approximation. " + debate_instr4
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": instruction4, "context": ["user query", thinking3.content, answer3.content], "agent_collaboration": "Debate"}
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for agent in debate_agents_4:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking3, answer3], instruction4, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking, answer = await agent(inputs, instruction4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking4[r].append(thinking)
            all_answer4[r].append(answer)
    final4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final4([taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1], "Subtask 4 decision: Given all the above thinking and answers, reason over them carefully and provide the final selected formula.", is_sub_task=True)
    sub_tasks.append(f"Subtask 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs