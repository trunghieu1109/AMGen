async def forward_6(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 0.1: Express r^2 in terms of a, b, c and relate a^2+b^2+c^2 to sigma1 and sigma2"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc0 = {"subtask_id": "subtask_0.1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking0, answer0 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id} analyzing r^2 expression, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0.1: ", sub_tasks[-1])
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction = "Sub-task 1.1: Formulate the problem of maximizing S = a^2+b^2+c^2 under constraints sigma2=27 and sigma3=23 via Lagrange multipliers" + debate_instr
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    subtask_desc1 = {"subtask_id": "subtask_1.1", "instruction": debate_instruction, "context": ["user query", thinking0.content, answer0.content], "agent_collaboration": "Debate"}
    all_thinking = [[]]
    all_answer = [[]]
    for agent in debate_agents:
        thinking1, answer1 = await agent([taskInfo, thinking0, answer0], debate_instruction, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
        all_thinking[0].append(thinking1)
        all_answer[0].append(answer1)
    final_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1f, answer1f = await final_decision([taskInfo, thinking0, answer0] + all_thinking[0] + all_answer[0], "Sub-task 1.1:" + final_instr, is_sub_task=True)
    agents.append(f"Final Decision agent calculating optimum case, thinking: {thinking1f.content}; answer: {answer1f.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking1f.content}; answer - {answer1f.content}")
    subtask_desc1['response'] = {"thinking": thinking1f, "answer": answer1f}
    logs.append(subtask_desc1)
    print("Step 1.1: ", sub_tasks[-1])
    sc_instruction = "Sub-task 2.1: Compute r^2=(a^2+b^2+c^2)/4 using the value from subtask 1.1, reduce to p/q and compute p+q"
    N = self.max_sc
    sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_think = []
    possible_ans = []
    subtask_desc2 = {"subtask_id": "subtask_2.1", "instruction": sc_instruction, "context": ["user query", thinking1f.content, answer1f.content], "agent_collaboration": "SC_CoT"}
    for ag in sc_agents:
        t2, a2 = await ag([taskInfo, thinking1f, answer1f], sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {ag.id}, thinking: {t2.content}; answer: {a2.content}")
        possible_think.append(t2)
        possible_ans.append(a2)
    final_decision2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision2([taskInfo, thinking1f, answer1f] + possible_think + possible_ans, "Sub-task 2.1: Synthesize and choose the most consistent answer for final result", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2.1: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking2, answer2, sub_tasks, agents)
    return final_answer, logs