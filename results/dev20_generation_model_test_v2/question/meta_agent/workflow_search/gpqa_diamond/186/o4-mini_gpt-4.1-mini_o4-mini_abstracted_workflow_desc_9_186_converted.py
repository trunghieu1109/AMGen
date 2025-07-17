async def forward_186(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot0_instruction = (
        "Sub-task 0: Extract and summarize key information from the query, "  
        "including ESPRESSO specs, detectability criterion (S/N>=10 in 1h), target stars, and answer choices."
    )
    cot0_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask0_desc = {"subtask_id": "subtask_0", "instruction": cot0_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking0, answer0 = await cot0_agent([taskInfo], cot0_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot0_agent.id}, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask0_desc['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask0_desc)
    print("Step 1: ", sub_tasks[-1])

    cot1_instruction = (
        "Sub-task 1: Compute apparent magnitudes for the M_V=15 stars at distances 5,10,50,200pc using m= M_V+5*log10(d/10), "  
        "and identify the faintest apparent magnitude reachable by ESPRESSO on an 8m VLT at S/N=10 in 1h."
    )
    N = self.max_sc
    cot1_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask1_desc = {"subtask_id": "subtask_1", "instruction": cot1_instruction, "context": ["user query", "thinking0", "answer0"], "agent_collaboration": "SC_CoT"}
    for agent in cot1_agents:
        thinking1_i, answer1_i = await agent([taskInfo, thinking0, answer0], cot1_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_decision1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth1_instr = "Sub-task 1: Synthesize and choose the most consistent apparent magnitudes and limiting m from above results."
    thinking1, answer1 = await final_decision1([taskInfo, thinking0, answer0] + possible_thinkings1 + possible_answers1, synth1_instr, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask1_desc['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask1_desc)
    print("Step 2: ", sub_tasks[-1])

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest estimation. Using insights, refine the S/N estimates for each star."
    critic_inst = "Please review the answer above and criticize where it might be wrong. If correct, output exactly 'True' in 'correct'."
    cot2_instruction = "Sub-task 2: Estimate the expected S/N for each star given its apparent magnitude, telescope area, throughput, exposure time and noise sources." + reflect_inst
    cot2_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask2_desc = {"subtask_id": "subtask_2", "instruction": cot2_instruction, "context": ["user query","thinking0","answer0","thinking1","answer1"], "agent_collaboration": "Reflexion"}
    thinking2, answer2 = await cot2_agent([taskInfo, thinking0, answer0, thinking1, answer1], cot2_instruction, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot2_agent.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking2, answer2], critic_inst, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        thinking2, answer2 = await cot2_agent([taskInfo, thinking0, answer0, thinking1, answer1, thinking2, answer2, feedback], cot2_instruction, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot2_agent.id}, refined thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask2_desc['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask2_desc)
    print("Step 3: ", sub_tasks[-1])

    debate_instr = "Given solutions from other agents, consider their advice and provide an updated count of detectable stars."
    debate_instruction = "Sub-task 3: Count how many stars satisfy S/N>=10 and select the correct multiple-choice answer." + debate_instr
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(self.max_round)]
    all_answer3 = [[] for _ in range(self.max_round)]
    subtask3_desc = {"subtask_id": "subtask_3", "instruction": debate_instruction, "context": ["user query","thinking2","answer2"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents:
            if r == 0:
                t3, a3 = await agent([taskInfo, thinking2, answer2], debate_instruction, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                t3, a3 = await agent(inputs, debate_instruction, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t3.content}; answer: {a3.content}")
            all_thinking3[r].append(t3)
            all_answer3[r].append(a3)
    final_decision3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking3, answer3 = await final_decision3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], final_instr3, is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask3_desc['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask3_desc)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs