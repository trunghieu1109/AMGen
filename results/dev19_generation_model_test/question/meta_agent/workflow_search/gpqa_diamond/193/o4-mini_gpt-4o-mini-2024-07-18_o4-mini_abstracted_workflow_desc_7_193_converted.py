async def forward_193(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    cot_sc_instruction_0 = "Sub-task 1: Generate the full list of 2^3 = 8 spin configurations for (S1, S2, S3), each taking values ±1."
    N0 = self.max_sc
    cot_agents_0 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N0)]
    possible_thinkings_0 = []
    possible_answers_0 = []
    subtask_desc_0 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction_0,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N0):
        thinking, answer = await cot_agents_0[i]([taskInfo], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0[i].id}, generating configurations, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings_0.append(thinking)
        possible_answers_0.append(answer)
    final_decision_agent_0 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + possible_thinkings_0 + possible_answers_0, "Sub-task 1: Synthesize and choose the most consistent list of configurations.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_0.id}, synthesizing configurations, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc_0["response"] = {"thinking":thinking0,"answer":answer0}
    logs.append(subtask_desc_0)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction_1 = "Sub-task 1: For each of the 8 configurations generated previously, compute the energy E = -J (S1·S2 + S1·S3 + S2·S3)."
    N1 = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings_1 = []
    possible_answers_1 = []
    subtask_desc_1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction_1,"context":["user query",thinking0.content,answer0.content],"agent_collaboration":"SC_CoT"}
    for i in range(N1):
        thinking, answer = await cot_agents_1[i]([taskInfo, thinking0, answer0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, computing energies, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings_1.append(thinking)
        possible_answers_1.append(answer)
    final_decision_agent_1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo, thinking0, answer0] + possible_thinkings_1 + possible_answers_1, "Sub-task 1: Synthesize and choose the most consistent energy calculations.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_1.id}, synthesizing energies, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1["response"] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc_1)
    print("Step 2: ", sub_tasks[-1])
    cot_sc_instruction_2 = "Sub-task 1: Group the configurations by their calculated energy values and count how many configurations share each energy (find degeneracy)."
    N2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings_2 = []
    possible_answers_2 = []
    subtask_desc_2 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction_2,"context":["user query",thinking0.content,answer0.content,thinking1.content,answer1.content],"agent_collaboration":"SC_CoT"}
    for i in range(N2):
        thinking, answer = await cot_agents_2[i]([taskInfo, thinking0, answer0, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, grouping degeneracies, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings_2.append(thinking)
        possible_answers_2.append(answer)
    final_decision_agent_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking0, answer0, thinking1, answer1] + possible_thinkings_2 + possible_answers_2, "Sub-task 1: Synthesize and choose the most consistent degeneracy counts.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_2.id}, synthesizing degeneracies, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2["response"] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc_2)
    print("Step 3: ", sub_tasks[-1])
    cot_instruction_3 = "Sub-task 1: Compute the partition function Z by summing over all distinct energy levels: Z = Σ_e (degeneracy of e) · exp(-β e)."
    cot_agent_3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking0, answer0, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, computing partition function, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3 = {"subtask_id":"subtask_1","instruction":cot_instruction_3,"context":["user query",thinking0.content,answer0.content,thinking1.content,answer1.content,thinking2.content,answer2.content],"agent_collaboration":"CoT"}
    subtask_desc_3["response"] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc_3)
    print("Step 4: ", sub_tasks[-1])
    debate_instruction = "Sub-task 2: Compare the derived expression for Z with the four provided choices and select the one that matches exactly. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking = [[] for _ in range(self.max_round)]
    all_answer = [[] for _ in range(self.max_round)]
    subtask_desc_4 = {"subtask_id":"subtask_2","instruction":debate_instruction,"context":["user query",thinking3.content,answer3.content],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking3, answer3], debate_instruction, r, is_sub_task=True)
            else:
                thinking, answer = await agent([taskInfo, thinking3, answer3] + all_thinking[r-1] + all_answer[r-1], debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing expressions, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    final_decision_agent_4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking3, answer3] + all_thinking[-1] + all_answer[-1], "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_4.id}, finalizing choice, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4["response"] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc_4)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs