async def forward_188(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction = "Sub-task 1: Extract and summarize the question, the four choices, and identify key physics terms ('effective particles', 'spontaneously-broken symmetry')."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc0 = {"subtask_id":"stage0_subtask1","instruction":cot_sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents:
        thinking0, answer0 = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_thinkings.append(thinking0)
        possible_answers.append(answer0)
    final_decision_agent0 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent0([taskInfo] + possible_thinkings + possible_answers, "Sub-task 1: Synthesize and choose the most consistent summary for extraction.", is_sub_task=True)
    sub_tasks.append(f"stage0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking":thinking0, "answer":answer0}
    logs.append(subtask_desc0)
    print("Step 1: ", sub_tasks[-1])

    debate_instruction1 = "Sub-task 1: Define clear criteria for association with a spontaneously-broken symmetry, invoking Goldstone’s theorem (massless modes from continuous symmetry breaking) versus topological excitations. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents1 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking1 = [[] for _ in range(N_max)]
    all_answer1 = [[] for _ in range(N_max)]
    subtask_desc1 = {"subtask_id":"stage1_subtask1","instruction":debate_instruction1,"context":["user query","thinking0","answer0"],"agent_collaboration":"Debate"}
    for r in range(N_max):
        for agent in debate_agents1:
            if r == 0:
                thinking1, answer1 = await agent([taskInfo, thinking0, answer0], debate_instruction1, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking0, answer0] + all_thinking1[r-1] + all_answer1[r-1]
                thinking1, answer1 = await agent(inputs, debate_instruction1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking1[r].append(thinking1)
            all_answer1[r].append(answer1)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo, thinking0, answer0] + all_thinking1[-1] + all_answer1[-1], "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"stage1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1, "answer":answer1}
    logs.append(subtask_desc1)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction2 = "Sub-task 1: Classify each candidate—Magnon, Pion, Phonon, Skyrmion—according to whether it is a Goldstone (or pseudo-Goldstone) mode of broken symmetry or a non-Goldstone topological excitation."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id":"stage2_subtask1","instruction":cot_sc_instruction2,"context":["user query","thinking1","answer1"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents2:
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings2.append(thinking2)
        possible_answers2.append(answer2)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2([taskInfo] + possible_thinkings2 + possible_answers2, "Sub-task 1: Synthesize and choose the most consistent classification.", is_sub_task=True)
    sub_tasks.append(f"stage2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2, "answer":answer2}
    logs.append(subtask_desc2)
    print("Step 3: ", sub_tasks[-1])

    debate_instruction3 = "Sub-task 1: From the classifications, identify which particle is not tied to a spontaneously-broken symmetry and justify why. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents3 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(N_max)]
    all_answer3 = [[] for _ in range(N_max)]
    subtask_desc3 = {"subtask_id":"stage3_subtask1","instruction":debate_instruction3,"context":["user query","thinking2","answer2"],"agent_collaboration":"Debate"}
    for r in range(N_max):
        for agent in debate_agents3:
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction3, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(inputs, debate_instruction3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"stage3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3, "answer":answer3}
    logs.append(subtask_desc3)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs