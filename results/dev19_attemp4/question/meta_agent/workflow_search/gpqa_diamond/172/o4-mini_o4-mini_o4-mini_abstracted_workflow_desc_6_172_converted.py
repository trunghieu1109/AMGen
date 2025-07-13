async def forward_172(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction = "Sub-task 1: Extract and organize given quantities electron mass, speed v, position uncertainty Δx, Planck constant ħ, and answer choices."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    possible_thinkings1 = []
    possible_answers1 = []
    for agent in cot_agents:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, extracting quantities, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings1.append(thinking)
        possible_answers1.append(answer)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    decision_instr1 = "Sub-task 1: Synthesize the most consistent extraction of given quantities."
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1, decision_instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction2 = "Sub-task 2: Based on extracted quantities, compute momentum uncertainty Δp = ħ/(2Δx) via Heisenberg uncertainty principle."
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction2,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"SC_CoT"}
    possible_thinkings2 = []
    possible_answers2 = []
    for agent in cot_agents2:
        thinking, answer = await agent([taskInfo,thinking1,answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, computing Δp, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings2.append(thinking)
        possible_answers2.append(answer)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    decision_instr2 = "Sub-task 2: Synthesize the most consistent momentum uncertainty calculation."
    thinking2, answer2 = await final_decision_agent2([taskInfo,thinking1,answer1] + possible_thinkings2 + possible_answers2, decision_instr2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction3 = "Sub-task 3: Compute the minimum energy uncertainty ΔE ≈ v * Δp using computed momentum uncertainty and given speed."
    cot_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_sc_instruction3,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"SC_CoT"}
    possible_thinkings3 = []
    possible_answers3 = []
    for agent in cot_agents3:
        thinking, answer = await agent([taskInfo,thinking2,answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, computing ΔE, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings3.append(thinking)
        possible_answers3.append(answer)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    decision_instr3 = "Sub-task 3: Synthesize the most consistent energy uncertainty calculation."
    thinking3, answer3 = await final_decision_agent3([taskInfo,thinking2,answer2] + possible_thinkings3 + possible_answers3, decision_instr3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction4 = "Sub-task 4: Compare numerical ΔE to provided choices and identify the closest estimate."
    cot_agents4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_sc_instruction4,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"SC_CoT"}
    possible_thinkings4 = []
    possible_answers4 = []
    for agent in cot_agents4:
        thinking, answer = await agent([taskInfo,thinking3,answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, comparing ΔE to choices, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings4.append(thinking)
        possible_answers4.append(answer)
    final_decision_agent4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    decision_instr4 = "Sub-task 4: Synthesize the most consistent estimate of energy uncertainty among choices."
    thinking4, answer4 = await final_decision_agent4([taskInfo,thinking3,answer3] + possible_thinkings4 + possible_answers4, decision_instr4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs